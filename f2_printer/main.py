import io
import asyncio
import json
import sys
import re
import traceback
import datetime
from aiohttp.client import ClientSession
from aiohttp.client_reqrep import ClientResponse
from PIL import Image
import img2pdf


class PrinterConfig:
    url_prefix: str
    access_key: str
    printer: str
    double_print_supported: bool
    def __init__(self, dict: dict) -> None:
        for key in dict:
            setattr(self, key, dict[key])

def retry(count: int, message: str):
    def decorator(func):
        async def inner(*args, **kwargs):
            for i in range(1, count + 1):
                try:
                    return await func(*args, **kwargs)
                except:
                    if i < count:
                        print(message.format(i))
                        await asyncio.sleep(i)
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

class PrinterClient:
    config: PrinterConfig
    access_headers: dict
    async_tasks: dict[str, asyncio.Task]
    incomplete_jobs: set[str]
    complete_jobs: set[str]
    pending_jobs: set[str]
    launched_jobs: set[str]
    pending_event: asyncio.Event
    jobs_changed_event: asyncio.Event
    def __init__(self, config: PrinterConfig) -> None:
        self.config = config
        self.access_headers = {
            "Authorization": "Bearer {}".format(config.get("access_key"))
        }
        self.async_tasks = {}
        self.pending_event = asyncio.Event()
        self.jobs_changed_event = asyncio.Event()
        self.incomplete_jobs = set()
        self.complete_jobs = set()
        self.pending_jobs = set()
        self.launched_jobs = set()

    async def ping_client(self, session: ClientSession):
        url = f'{self.config.get("url_prefix")}ping'
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
                    print(f'There are {len(self.incomplete_jobs)} incomplete jobs and {len(self.complete_jobs)} complete jobs.')
                if job_change:
                    self.jobs_changed_event.set()







    async def send_to_printer(self, id: str, bytes: bytes) -> str | None:
        params = [
            'lp',
            '-d', self.config.get("printer"),
            '-o', 'fit-to-page',
            '-o', 'media=custom_53.62x85.37mm_53.62x85.37mm',
            '-o', 'sides=two-sided-short-edge',
            '-t', id
        ]
        pdf = "pdf" in self.config.get("printer") or "PDF" in self.config.get("printer")
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

        print(f'Out: {out}')
        if proc.returncode == 0:
            job_id = out.split(" ")[3]
            if pdf:
                tid = f'{job_id}_wait'
                async def release():
                    print(f'PDF {job_id} - sleeping for 10 seconds before release')
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

    async def watch_job(self, print_job_id, job_id, item_id):
        begin = datetime.datetime.now(datetime.timezone.utc)
        max_time = begin + datetime.timedelta(minutes=1)
        while datetime.datetime.now(datetime.timezone.utc) < max_time:
            await self.jobs_changed_event.wait()
            if print_job_id in self.complete_jobs:
                # Remove from the list
                if print_job_id in self.pending_jobs:
                    print(f'Acknowledged that {print_job_id} is complete for job {job_id} - item {item_id}')
                    self.pending_jobs.remove(print_job_id)
                return True
            await asyncio.sleep(0.1)

        print(f'Timed out on print job for {print_job_id}')
        return False

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
        print_job_id = await self.send_to_printer(item.id, bytes)
        job_success = await self.watch_job(print_job_id, job, item.id)
        if job_success:
            success = await self.item_success(session, item.id)
            if not success:
                print("Could not mark item as successful")
            else:
                print("Item successful")
        else:
            failure = await self.item_failed(session, item.id)
            if not failure:
                print("Could not mark item as failed")
            else:
                print("Item failed")

    async def print_job(self, session: ClientSession, job: str):
        while True:
            print("Polling job {}".format(job))
            item_response = await self.poll_item(session, job)
            if item_response.item:
                await self.print_item(session, job, item_response.item)
            if item_response.wait and item_response.wait > 0:
                await asyncio.sleep(float(item_response.wait))
            if not item_response.poll_again:
                break

    async def run(self):
        watch_printer_task = asyncio.get_event_loop().create_task(self.watch_jobs())
        async with ClientSession(read_timeout=5, conn_timeout=5) as session:
            if not await self.ping_client(session):
                print("Could not authenticate. Check the access_key")
                return
            while True:
                job = await self.poll_job(session)
                if not job:
                    await asyncio.sleep(1)
                    continue
                await self.print_job(session, job)
        # Not that under normal conditions we get here, but for the sake of purity
        watch_printer_task.cancel()




async def run(config: PrinterConfig):
    client = PrinterClient(config)
    await client.run()


def main():
    config = {}
    if len(sys.argv) == 1:
        print("Specify configuration file")

    with io.open(sys.argv[1], "r") as f:
        config = json.load(f)

    if not config or len(config) == 0:
        print("Configuration not right")
        exit(1)
    asyncio.get_event_loop().run_until_complete(run(config))


if __name__ == "__main__":
    main()
