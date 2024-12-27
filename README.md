# F2 Printer

A utility to print badges to a CR80 printer from a queue system

Also converts partial document-ir to ESC/POS


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
    "columns": 48
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

With Dev Containers, `cups-pdf` is installed, so you should be able to use that.
Files will appear in the `pdf` folder.

Change the URL and access key as needed and save to `config/print.json`

```json
{
  "FirstPrinter": {
    "url_prefix": "http://192.168.1.100:8000/registration/printing/",
    "access_key": "ABCDwhateverwhatever",
    "printer": "cups-pdf",
    "thermal": true,
    "columns": 48
  },
  "SecondPrinter": {
    "url_prefix": "http://192.168.1.100:8000/registration/printing/",
    "access_key": "ABCDwhateverwhatever",
    "printer": "cups-pdf",
    "double_print_supported": true,
    "cr80": true
  }
}
```

And finally to run it, use:

```bash
./start.sh
```

## Operate

To set up the machine, use `root-install.sh` on a fresh debian install.
Run that as root.

You may want to install the driver for the printer, which can be found at:
https://www.hidglobal.com/drivers/41707

It does list some really old 32 bit linux systems, it actually still works fine on 64 bit.

Make sure to set up the printer in the cups admin at https://localhost:631 (ignore the cert error),
and to select the correct ribbon type that is inserted.

A handy script called `start.sh` is ready for you.
It expects all the printers are set up (as described above) in `config/print.json`.
