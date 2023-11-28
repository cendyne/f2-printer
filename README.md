# F2 Printer

A utility to print badges to a CR80 printer from a queue system

Also converts partial document-ir to ESC/POS

## Setup

Run `poetry run python3 setup.py`

## Local development

https://localhost:8631
Under Administration, user `root`, password `root`
You can add printers, but you'll need to use the IP address of the host instead of a `.local` address.
It'll look something like `ipp://192.168.1.100:631/printers/NameHere`

You'll want to install dependencies with

```
poetry install
```

And then make a folder called `config`

```bash
mkdir config
```

and create a config inside with a name like `config/print.json` and its contents could be...

For example a thermal printer

```json
{
  "FirstPrinter": {
    "url_prefix": "http://192.168.1.100:8000/registration/printing/",
    "access_key": "ABCDwhateverwhatever",
    "printer": "NameHere",
    "thermal": true,
    "columns": 80
  }
}
```

For example a CR80 printer

```json
{
  "SecondPrinter": {
    "url_prefix": "http://192.168.1.160:8000/registration/printing/",
    "access_key": "ABCDwhateverwhatever",
    "printer": "NameHere",
    "double_print_supported": true,
    "cr80": true
  }
}
```

And finally to run, do something like...
```
poetry run print config/print.json
```

Note that the `print.json` file would be the same name as you did above.

## Operate

To set up the machine, use `root-install.sh` on a fresh debian install.
Run that as root.

A handy script called `start.sh` is ready for you.
It expects all the printers are set up (as described above) in `config/print.json`.
