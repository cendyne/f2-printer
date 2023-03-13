import io
import asyncio
import json
import sys
import re
import traceback
import datetime
import urllib.parse
from enum import Enum
from aiohttp.client import ClientSession
from aiohttp.client_reqrep import ClientResponse
from PIL import Image
import img2pdf
import random


class PrinterConfig:
    url_prefix: str
    access_key: str
    printer: str
    double_print_supported: bool

    def __init__(self, dict: dict) -> None:
        for key in dict:
            setattr(self, key, dict[key])

    def get(self, key: str):
        return getattr(self, key)


class PrinterSetConfig:
    configs: dict[str, PrinterConfig]
    _keys: set[str]

    def __init__(self, dict) -> None:
        self._keys = set()
        for key in dict:
            setattr(self, key, PrinterConfig(dict[key]))
            self._keys.add(key)

    def keys(self) -> set[str]:
        return self._keys

    def __getitem__(self, item):
        return getattr(self, item)


def retry(count: int, message: str):
    def decorator(func):
        async def inner(*args, **kwargs):
            for i in range(1, count + 1):
                try:
                    return await func(*args, **kwargs)
                except:
                    if i < count:
                        print(message.format(i))
                        await asyncio.sleep(min(0.001, i + random.uniform(-0.1, 0.5)))
                        continue
                    raise
        return inner
    return decorator


class ItemResponseItem:
    id: str
    double_sided: bool
    label: str

    def __init__(self, id, double_sided, label) -> None:
        self.id = id
        self.double_sided = double_sided or False
        self.label = label or id


class ItemResponse:
    item: ItemResponseItem | None
    wait: int
    poll_again: bool

    def __init__(self, item, wait, poll_again) -> None:
        self.item = item
        self.wait = wait
        self.poll_again = poll_again


class PrinterStatus(Enum):
    DISABLED = 1
    IDLE = 2
    PRINTING = 3
    UNKNOWN = 4


class PrinterApi:
    pending_event: asyncio.Event
    jobs_changed_event: asyncio.Event
    incomplete_jobs: set[str]
    complete_jobs: set[str]
    pending_jobs: set[str]
    launched_jobs: set[str]
    status: dict[str, PrinterStatus]
    active_jobs: set[str]
    active_jobs_kv: dict[str, str]
    async_tasks: dict[str, asyncio.Task]
    last_print: datetime.datetime | None
    disabled_reasons: dict[str, str]
    printing_reasons: dict[str, str]

    def __init__(self) -> None:
        self.pending_event = asyncio.Event()
        self.jobs_changed_event = asyncio.Event()
        self.incomplete_jobs = set()
        self.complete_jobs = set()
        self.pending_jobs = set()
        self.launched_jobs = set()
        self.status = {}
        self.active_jobs = set()
        self.active_jobs_kv = dict()
        self.async_tasks = {}
        self.last_print = None
        self.disabled_reasons = {}
        self.printing_reasons = {}
        pass

    async def watch_jobs(self) -> None:
        while True:
            print("Waiting for pending event")
            await self.pending_event.wait()
            print("Got pending event!")
            self.pending_event.clear()
            wait = False
            count = 0
            while len(self.pending_jobs) > 0:
                if wait:
                    await asyncio.sleep(0.25)
                self.jobs_changed_event.clear()
                # Next loop should wait
                noisy = count % 20 == 0
                wait = True
                count += 1
                # if noisy:
                #     print(f'Watching print jobs with lpstat with {len(self.pending_jobs)} remaining jobs')
                complete_jobs = set()
                incomplete_jobs = set()
                # print("lpstat -W not-completed")
                params = [
                    'lpstat',
                    '-W', 'not-completed'
                ]
                proc = await asyncio.create_subprocess_exec(*params,
                                                            shell=False,
                                                            stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.PIPE
                                                            )
                # print(f'Created sub process {proc.pid}')
                stdout, stderr = await proc.communicate()
                # print("Collected output")
                for line in stdout.decode("utf-8").split("\n"):
                    parts = re.split('\s+', line)
                    job_id = parts[0]
                    if job_id in self.launched_jobs:
                        incomplete_jobs.add(job_id)
                # print("lpstat -W completed")
                params = [
                    'lpstat',
                    '-W', 'completed'
                ]
                proc = await asyncio.create_subprocess_exec(*params,
                                                            shell=False,
                                                            stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.PIPE
                                                            )
                stdout, stderr = await proc.communicate()
                for line in stdout.decode("utf-8").split("\n"):
                    parts = re.split('\s+', line)
                    job_id = parts[0]
                    if job_id in self.launched_jobs:
                        complete_jobs.add(job_id)

                job_change = False
                if len(self.incomplete_jobs) != len(incomplete_jobs) or len(self.complete_jobs) != len(complete_jobs):
                    job_change = True

                self.incomplete_jobs = incomplete_jobs
                self.complete_jobs = complete_jobs
                if noisy:
                    print(
                        f'There are {len(self.incomplete_jobs)} incomplete jobs and {len(self.complete_jobs)} complete jobs.')
                if job_change:
                    self.jobs_changed_event.set()

    def parse_status(self, line: str) -> tuple[str, PrinterStatus, str | None, str | None] | None:
        parts = re.split("\s+", line)

        if parts[0] == 'printer':
            printer = parts[1]
            # print(f'Parts: {printer} {parts[2:]}')
            if parts[2] == 'disabled':
                reason = "Unplugged or turned off"
                for i in range(3, len(parts) + 1):
                    if parts[i] == '-':
                        reason = " ".join(parts[i+1:])
                        break
                return (printer, PrinterStatus.DISABLED, reason, None)
            if parts[2] == 'is' and parts[3] == 'idle.' and parts[4] == 'enabled':
                return (printer, PrinterStatus.IDLE, None)
            if parts[2] == 'now' and parts[3] == 'printing':
                job = parts[4]
                if job.endswith("."):
                    job = job[0:-1]
                timezone_part = 5
                for i in range(5, len(parts)):
                    t = parts[i]
                    if re.match("[CMPEH][AO]?[DS]?T|AK|UTC|GMT", t):
                        timezone_part = i
                        break
                status = " ".join(parts[timezone_part+1:])
                return (printer, PrinterStatus.PRINTING, job, status)
            return (printer, PrinterStatus.UNKNOWN, None, None)
        return None

    async def poll_status(self):
        status = {}
        active_jobs = set()
        active_jobs_kv = dict()
        disabled_reasons = {}
        printing_reasons = {}
        params = [
            'lpstat',
            '-p'
        ]
        proc = await asyncio.create_subprocess_exec(*params,
                                                    shell=False,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE
                                                    )

        stdout, stderr = await proc.communicate()
        str_output = stdout.decode("utf-8")
        last_line = ""
        for line in str_output.split("\n"):
            if line.startswith("\t"):
                last_line += line
                continue
            if last_line == "":
                last_line = line
                pass
            result = self.parse_status(last_line)
            if result:
                status[result[0]] = result[1]
                if result[1] == PrinterStatus.PRINTING and result[2]:
                    active_jobs.add(result[2])
                    active_jobs_kv[result[0]] = result[2]
                    if result[3]:
                        printing_reasons[result[0]] = result[3]
                if result[1] == PrinterStatus.DISABLED and result[2]:
                    disabled_reasons[result[0]] = result[2]
            last_line = line
        if last_line != "":
            result = self.parse_status(last_line)
            if result:
                status[result[0]] = result[1]
                if result[1] == PrinterStatus.PRINTING and result[2]:
                    active_jobs.add(result[2])
                    active_jobs_kv[result[0]] = result[2]
                    if result[3]:
                        printing_reasons[result[0]] = result[3]
                if result[1] == PrinterStatus.DISABLED and result[2]:
                    disabled_reasons[result[0]] = result[2]
        self.status = status
        self.active_jobs = active_jobs
        self.active_jobs_kv = active_jobs_kv
        self.disabled_reasons = disabled_reasons
        self.printing_reasons = printing_reasons
        # Maybe push an event
        # print(f'Polled! {status}')

    async def send_to_printer(self, printer: str, id: str, bytes: bytes) -> str | None:
        params = [
            'lp',
            '-d', printer,
            '-o', 'fit-to-page',
            '-o', 'media=custom_53.62x85.37mm_53.62x85.37mm',
            '-o', 'sides=two-sided-short-edge',
            '-t', id
        ]
        pdf = "pdf" in printer or "PDF" in printer
        if pdf:
            print('')
            params.append('-H')
            params.append('indefinite')

        proc = await asyncio.create_subprocess_exec(*params,
                                                    shell=False,
                                                    stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE
                                                    )
        stdout, stderr = await proc.communicate(bytes)
        print(f'[lp exited with {proc.returncode}]')
        out = stdout.decode("utf-8").split("\n")[0]

        self.last_print = datetime.datetime.now(datetime.timezone.utc)

        print(f'Out: {out}')
        if proc.returncode == 0:
            job_id = out.split(" ")[3]
            if pdf:
                tid = f'{job_id}_wait'

                async def release():
                    print(
                        f'PDF {job_id} - sleeping for 10 seconds before release')
                    await asyncio.sleep(10)
                    if tid in self.async_tasks:
                        self.async_tasks.pop(tid)
                    params = [
                        'lp',
                        '-i', job_id,
                        '-H', 'release'
                    ]
                    proc = await asyncio.create_subprocess_exec(*params, shell=False)
                    result = await proc.wait()
                    print(f'lp release {job_id} - {result}')
                task = asyncio.get_event_loop().create_task(release())
                self.async_tasks[tid] = task
            self.launched_jobs.add(job_id)
            self.pending_jobs.add(job_id)
            self.pending_event.set()
            return job_id
        else:
            err = stderr.decode("utf-8")
            print(f'Err: {err}')
            return None

    async def cancel_job(self, job_id) -> bool:
        params = [
            'cancel',
            job_id
        ]

        proc = await asyncio.create_subprocess_exec(*params,
                                                    shell=False,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE
                                                    )
        stdout, stderr = await proc.communicate()
        if proc.returncode > 0:
            print(stdout.decode("utf-8"))
            print(stderr.decode("utf-8"))
        if proc.returncode == 0:
            self.pending_jobs.remove(job_id)
            return True
        return False

    async def watch_job(self, printer, print_job_id, job_id, item_id):
        begin = datetime.datetime.now(datetime.timezone.utc)
        max_time = begin + datetime.timedelta(seconds=90)
        reason = None
        while datetime.datetime.now(datetime.timezone.utc) < max_time:
            if print_job_id in self.complete_jobs:
                # Remove from the list
                if print_job_id in self.pending_jobs:
                    print(
                        f'Acknowledged that {print_job_id} is complete for job {job_id} - item {item_id}')
                    self.pending_jobs.remove(print_job_id)
                return True
            if printer in self.printing_reasons:
                latest_reason = self.printing_reasons[printer]
            else:
                latest_reason = None
            if latest_reason != reason:
                print(f'Print job {print_job_id} changed from {reason} to {latest_reason}')
                reason = latest_reason
                if reason and ("Error" in reason or "Card Jam" in reason):
                    return False
            await asyncio.sleep(0.1)

        print(f'Timed out on print job for {print_job_id}')
        return False

    async def uses_systemd(self) -> bool:
        found_systemd = False
        params = [
            'ps',
            'aux'
        ]
        proc = await asyncio.create_subprocess_exec(*params,
                                                    shell=False,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE
                                                    )
        stdout, stderr = await proc.communicate()
        for line in stdout.decode("utf-8").split("\n"):
            if "systemd" in line:
                found_systemd = True
                break
        return found_systemd

    async def restart_cups(self) -> bool:
        params = []
        if await self.uses_systemd():
            # Requires the following in sudoers
            # %lpadmin ALL= NOPASSWD: /usr/bin/systemctl restart cups.service
            params = [
                'sudo',
                'systemctl',
                'restart',
                'cups.service'
            ]
        else:
            # This is running in a container, probably
            params = [
                'sudo',
                'service',
                'restart',
                'cups'
            ]

        proc = await asyncio.create_subprocess_exec(*params,
                                                    shell=False,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE
                                                    )
        stdout, stderr = await proc.communicate()
        if proc.returncode > 0:
            print(stdout.decode("utf-8"))
            print(stderr.decode("utf-8"))
        return proc.returncode == 0

    async def poll_background(self):
        while True:
            # print(f'Polling status')
            await self.poll_status()
            await asyncio.sleep(0.25)

    async def run(self):
        bg = asyncio.get_event_loop().create_task(self.poll_background())
        watch_printer_task = asyncio.get_event_loop().create_task(self.watch_jobs())
        print(f'Printer API running!')
        await asyncio.gather(bg, watch_printer_task)


class PrinterClient:
    client_name: str
    config: PrinterConfig
    access_headers: dict
    session: ClientSession
    api: PrinterApi
    last_ping: datetime.datetime | None
    waiting_for_disconnect: bool
    waiting_for_reconnect: bool
    waiting_for_cups_restart: bool
    holding: bool
    current_print_job: str | None
    unexpected_disconnect: bool

    def __init__(self, client_name: str, config: PrinterConfig, session: ClientSession, api: PrinterApi) -> None:
        self.client_name = client_name
        self.config = config
        self.access_headers = {
            "Authorization": "Bearer {}".format(config.get("access_key"))
        }

        self.session = session
        self.api = api
        self.last_ping = None
        self.waiting_for_disconnect = False
        self.waiting_for_reconnect = False
        self.waiting_for_cups_restart = False
        self.current_print_job = None
        self.unexpected_disconnect = False
        self.holding = False

    async def ping_client(self, session: ClientSession, message: str | None = None):
        url = f'{self.config.get("url_prefix")}ping'
        if message:
            url += "?message=" + urllib.parse.quote(message)
        success = False
        try:
            async with session.get(url, headers=self.access_headers) as resp:
                r: ClientResponse = resp
                if r.status == 200:
                    success = True
        except:
            pass
        return success

    async def poll_job(self, session: ClientSession) -> str | None:
        url = f'{self.config.get("url_prefix")}poll-job'
        try:
            async with session.post(url, headers=self.access_headers) as resp:
                r: ClientResponse = resp
                if r.status != 200:
                    return None
                json = await r.json()
                if "job" in json:
                    return json["job"]
                return None
        except:
            print("Could not poll job, network error maybe?")
            return None

    async def hold_job(self, session: ClientSession, job: str) -> str | None:
        url = f'{self.config.get("url_prefix")}hold-job/{job}'
        try:
            async with session.post(url, headers=self.access_headers) as resp:
                r: ClientResponse = resp
                if r.status != 200:
                    return False
                json = await r.json()
                if "received" in json:
                    return json["received"]
                return False
        except:
            print("Could not hold job, network error maybe?")
            return False

    async def release_job(self, session: ClientSession, job: str) -> str | None:
        url = f'{self.config.get("url_prefix")}release-job/{job}'
        try:
            async with session.post(url, headers=self.access_headers) as resp:
                r: ClientResponse = resp
                if r.status != 200:
                    return False
                json = await r.json()
                if "received" in json:
                    return json["received"]
                return False
        except:
            print("Could not release job, network error maybe?")
            return False

    async def poll_item(self, session: ClientSession, job: str) -> ItemResponse | None:
        poll_item = f'{self.config.get("url_prefix")}poll-item/{job}'
        try:
            async with session.post(poll_item, headers=self.access_headers) as resp:
                r: ClientResponse = resp
                if r.status != 200:
                    return None
                json = await r.json()
                item = None
                if not "poll_again" in json:
                    return None
                if "item" in json and not json["item"] is None:
                    item = ItemResponseItem(
                        json["item"]["id"], json["item"]["double_sided"], json["item"]["label"])
                return ItemResponse(item, json["wait"], json["poll_again"])
        except:
            print("Could not poll item, network error maybe?")
            # Application will automatically poll again if we tell it to.
            return ItemResponse(None, 1, True)

    @retry(10, "Could not download item frontside, trying again after {} seconds")
    async def item_frontside(self, session: ClientSession, item: str) -> bytes | None:
        url = f'{self.config.get("url_prefix")}item-frontside/{item}'
        async with session.get(url, headers=self.access_headers) as resp:
            r: ClientResponse = resp
            if r.status != 200:
                return None
            return await r.read()

    @retry(10, "Could not download item backside, trying again after {} seconds")
    async def item_backside(self, session: ClientSession, item: str) -> bytes | None:
        url = f'{self.config.get("url_prefix")}item-backside/{item}'
        async with session.get(url, headers=self.access_headers) as resp:
            r: ClientResponse = resp
            if r.status != 200:
                return None
            return await r.read()

    @retry(10, "Could not report item success, trying again after {} seconds")
    async def item_success(self, session: ClientSession, item: str) -> None | bool:
        url = f'{self.config.get("url_prefix")}item-succeeded/{item}'
        async with session.post(url, headers=self.access_headers) as resp:
            r: ClientResponse = resp
            if r.status != 200:
                return None
            json = await r.json()
            if "received" in json:
                return json["received"]
            return None

    @retry(10, "Could not report item failed, trying again after {} seconds")
    async def item_failed(self, session: ClientSession, item: str) -> None | bool:
        url = f'{self.config.get("url_prefix")}item-failed/{item}'
        async with session.post(url, headers=self.access_headers) as resp:
            r: ClientResponse = resp
            if r.status != 200:
                return None
            json = await r.json()
            if "received" in json:
                return json["received"]
            return None

    async def report_failure(self, session: ClientSession, item_id: str):
        failure = await self.item_failed(session, item_id)
        if not failure:
            print("Could not mark item as failed")
        else:
            print("Item failed")

    async def report_success(self, session: ClientSession, item_id: str):
        success = await self.item_success(session, item_id)
        if not success:
            print("Could not mark item as successful")
        else:
            print("Item successful")

    async def print_item(self, session: ClientSession, job: str, item: ItemResponseItem):
        print("Polled item {}".format(item.label))
        images = []
        front = await self.item_frontside(session, item.id)
        if front:
            images.append(front)
            print("Got front {}".format(item.id))
        if item.double_sided:
            backside = await self.item_backside(session, item.id)
            if backside:
                images.append(backside)
                print("Got back {}".format(item.id))
        print("Got {} images".format(len(images)))
        stream = io.BytesIO()
        img2pdf.convert(*images, outputstream=stream)
        stream.seek(0)
        bytes = stream.read()
        await self.ping_client(session, "Sending to printer")
        self.current_print_job = await self.api.send_to_printer(self.config.printer, item.id, bytes)
        if not self.current_print_job:
            await self.report_failure(session, item.id)
            return
        job_success = await self.api.watch_job(self.config.printer, self.current_print_job, job, item.id)
        if job_success:
            await self.report_success(session, item.id)
        else:
            # Cancel job
            if not await self.api.cancel_job(self.current_print_job):
                print("Spicy, can't cancel the job")
            await self.report_failure(session, item.id)
            # This printer is misbehaving
            self.waiting_for_disconnect = True
            await self.ping_client(session, "Printer needs to be reset")
        self.current_print_job = None

    async def hold_job_for_restart(self, session: ClientSession, job: str):
        # This process should take at most a minute right?
        # But for robustness, wait at most 2 minutes before
        # assuming something went wrong
        # And technically we could use an event here and gather
        # But this seems more straight forward in the short term.
        self.holding = True
        now = datetime.datetime.now(datetime.timezone.utc)
        next_hold = now + datetime.timedelta(seconds=30)
        max_time = now + datetime.timedelta(minutes=2)
        while now < max_time:
            now = datetime.datetime.now(datetime.timezone.utc)
            if not self.waiting_for_cups_restart:
                break
            if now > next_hold:
                # Keep the job pinned to this client + printer
                next_hold = now + \
                    datetime.timedelta(
                        seconds=20) + datetime.timedelta(milliseconds=random.randint(-800, 800))
                await self.hold_job(session, job)
            # Wait half a second and try again
            await asyncio.sleep(0.5)

        if self.waiting_for_cups_restart:
            print(
                f'Print client {self.client_name} waited too long and will resume printing')
            self.waiting_for_cups_restart = False
        self.holding = False

    async def print_job(self, session: ClientSession, job: str):
        while True:
            if self.waiting_for_disconnect:
                # Release this job
                await self.release_job(session, job)
                reason = 'Please disconnect'
                if self.config.printer in self.api.printing_reasons:
                    reason = self.api.printing_reasons[self.config.printer]
                elif self.config.printer in self.api.disabled_reasons:
                    reason = self.api.disabled_reasons[self.config.printer]
                if not reason:
                    reason = 'Please turn off and on again'
                if reason and 'unknown' in reason:
                    reason = 'Please turn off and on again'
                await self.ping_client(self.session, f'Problem: {reason}')
                break
            print("Polling job {}".format(job))
            item_response = await self.poll_item(session, job)
            if item_response.item:
                await self.print_item(session, job, item_response.item)
            if item_response.wait and item_response.wait > 0:
                await asyncio.sleep(float(item_response.wait))
            if not item_response.poll_again:
                break
            if self.waiting_for_cups_restart:
                await self.hold_job_for_restart(session, job)

    async def run(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        printer = self.config.printer
        printer_status = self.api.status[printer]
        if self.waiting_for_disconnect:
            if printer_status == PrinterStatus.DISABLED:
                # It appears disconnected now!
                self.waiting_for_disconnect = False
                self.waiting_for_reconnect = True
                print(f'Printer {printer} disconnected')
                await self.ping_client(self.session, 'Disconnected, waiting for reconnect')
            return
        if self.waiting_for_reconnect:
            if printer_status == PrinterStatus.IDLE:
                print(f'Printer {printer} reconnected, waiting for cups restart')
                self.waiting_for_reconnect = False
                self.waiting_for_cups_restart = True
                await self.ping_client(self.session, 'Reconnected, waiting for cups restart')
            elif printer_status == PrinterStatus.PRINTING:
                print(f'Printer {printer} reconnected with status Printing, that is odd')
                # That's weird, it should be free unless there's a job jam or something.
                active_job = self.api.active_jobs_kv[printer]
                if active_job:
                    print(
                        f'Printer {printer} found to be printing {active_job}. A surprise. Canceling.')
                    if not await self.api.cancel_job(active_job):
                        print(
                            f'Tried to cancel job {active_job} on printer {printer}, but failed.')
            elif printer_status == PrinterStatus.DISABLED:
                reason = self.api.disabled_reasons[printer]
                print(f'Printer {printer} reconnected with disabled status: {reason}')
                if reason and "unknown" in reason:
                    await self.ping_client(self.session, f'Turn off printer by the power button and turn it back on')
                else:
                    await self.ping_client(self.session, f'Disabled: {reason or "Unknown reason"}')
                self.holding = True
                await asyncio.sleep(random.uniform(5.0, 10.0))
                self.holding = False
            else:
                print(f'Reconnected {printer} with unexpected status {printer_status}')
            return
        if self.waiting_for_cups_restart:
            print(f'Printer {printer} waiting for cups restart')
            await self.ping_client(self.session, 'Waiting for cups restart')
            self.holding = True
            await asyncio.sleep(random.uniform(2.0, 5.0))
            self.holding = False
            # Skipping
            return
        if not self.unexpected_disconnect and printer_status == PrinterStatus.DISABLED:
            await asyncio.sleep(0.15 + random.uniform(-0.1, 0.5))
            await self.ping_client(self.session, 'Disconnected')
            self.unexpected_disconnect = True
            return
        if self.unexpected_disconnect:
            if printer_status != PrinterStatus.DISABLED:
                self.unexpected_disconnect = False
                await asyncio.sleep(0.15 + random.uniform(-0.1, 0.5))
                await self.ping_client(self.session, 'Reconnected')
            else:
                # No need to probe further
                return
        if not self.last_ping or self.last_ping < now - datetime.timedelta(minutes=1):
            if not await self.ping_client(self.session):
                print(
                    f'Could not authenticate. Check the access_key {self.config.access_key[0:8]}...{self.config.access_key[-8:]}')
                return
            self.last_ping = now
            await asyncio.sleep(1 + random.uniform(-0.1, 0.5))
        if printer_status == PrinterStatus.DISABLED:
            return
        job = await self.poll_job(self.session)
        if not job:
            await asyncio.sleep(1 + random.uniform(-0.1, 0.5))
            # await asyncio.sleep(1)
            return
        await self.print_job(self.session, job)


class PrintManagement:
    session: ClientSession
    config: PrinterSetConfig
    client_name: str
    hold: set[str]
    waiting_for_cups_restart: bool

    def __init__(self, config: PrinterSetConfig) -> None:
        self.session = ClientSession(read_timeout=5, conn_timeout=5)
        self.hold = set()
        self.config = config
        self.waiting_for_cups_restart = False

    async def run_client(self, key: str, client: PrinterClient):
        try:
            self.hold.add(key)
            await client.run()
        except Exception as e:
            print(f'Error in running job? {e}')
            traceback.print_exception(e)
        finally:
            self.hold.remove(key)

    async def restart_procedure(self, clients: dict[str, PrinterClient], api: PrinterApi):
        for k in clients.keys():
            client = clients[k]
            client.waiting_for_cups_restart = True
        now = datetime.datetime.now(datetime.timezone.utc)
        # Wait at most 70 seconds for pending print items to complete
        # Expect that items will self cancel after 60 seconds
        max_wait = now + datetime.timedelta(seconds=70)
        while datetime.datetime.now(datetime.timezone.utc) < max_wait:
            # Check if all clients are finished
            holding = 0
            for k in clients.keys():
                client = clients[k]
                if client.holding:
                    holding += 1
            if len(self.hold) == holding:
                print(f'All {holding} clients are holding, we may proceed and restart')
                break
            # Yield and give time to print processes to realize they should hold
            await asyncio.sleep(0.25)
        await api.restart_cups()
        # Now that cups is restarted, clear waiting for cups
        for k in clients.keys():
            client = clients[k]
            if client.waiting_for_cups_restart:
                client.waiting_for_cups_restart = False
        # Yield and give time to print processes to realize they should resume
        await asyncio.sleep(0.25)

    async def run_clients(self, clients: dict[str, PrinterClient], api: PrinterApi):
        for k in clients.keys():
            client = clients[k]
            if client.waiting_for_cups_restart:
                self.waiting_for_cups_restart = True
                break
            now = datetime.datetime.now(datetime.timezone.utc)
            if k in self.hold:
                # The client is still running
                continue

            printer = client.config.printer
            if not printer in api.status:
                print(f'Printer missing! {printer} {api.status}')
                continue
            # Asynchronously prod the client to run
            asyncio.get_event_loop().create_task(self.run_client(k, client))
            # Slow things down when there's no work to do
            if not api.last_print or api.last_print < now - datetime.timedelta(minutes=5):
                await asyncio.sleep(10)
                continue

    async def run(self):
        api = PrinterApi()
        # Run the printer API in the background
        asyncio.get_event_loop().create_task(api.run())
        clients: dict[str, PrinterClient] = dict()
        for k in self.config.keys():
            clients[k] = PrinterClient(k, self.config[k], self.session, api)
            print(f'Initialized client {k}')
        while True:
            await asyncio.sleep(0.1)
            if len(api.status) > 0:
                break
        print(f'Activating with {api.status}')
        while True:
            await asyncio.sleep(0.1)
            self.waiting_for_cups_restart = False
            await self.run_clients(clients, api)
            if self.waiting_for_cups_restart:
                await self.restart_procedure(clients, api)


async def run(config: PrinterConfig):
    management = PrintManagement(config)
    await management.run()


def main():

    if len(sys.argv) == 1:
        print("Specify configuration file")

    config = {}
    with io.open(sys.argv[1], "r") as f:
        config = json.load(f)

    if not config or len(config) == 0:
        print("Configuration not right")
        exit(1)
    config = PrinterSetConfig(config)

    asyncio.get_event_loop().run_until_complete(run(config))


if __name__ == "__main__":
    main()
