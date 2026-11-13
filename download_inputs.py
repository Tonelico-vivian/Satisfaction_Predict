"""Download only the public artifacts needed by the 245-member method."""
import argparse
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from common import ROOT, read_json, save_json

API = "https://www.kaggle.com/api/v1/"


def request(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/public")
    args = parser.parse_args()
    catalog = read_json(ROOT / "metadata/public_sources.json")
    records = []
    for tag, source in catalog.items():
        urls = {}
        if source["kind"] == "kernel":
            owner, slug = source["ref"].split("/")
            query = urllib.parse.urlencode({"userName": owner, "kernelSlug": slug, "pageSize": 200})
            with request(API + "kernels/output?" + query) as stream:
                output = json.load(stream)
            assert not output.get("nextPageToken"), tag
            for item in output["files"]:
                if Path(item["fileName"]).name in source["files"]:
                    session = item["url"].split("/kf/")[1].split("/")[0]
                    assert session == source["session"], (
                        f"{tag}: public output session changed. Obtain pinned session {source['session']} "
                        "instead of silently substituting a different model.")
                    urls[Path(item["fileName"]).name] = item["url"]
        for name in source["files"]:
            if source["kind"] == "dataset":
                url = API + "datasets/download/" + source["ref"] + "/" + urllib.parse.quote(name, safe="/")
                url += "?datasetVersionNumber=" + str(source["version"])
            else:
                url = urls[name]
            target = args.output_dir / tag / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + ".partial")
            with request(url) as response, temp.open("wb") as stream:
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError(f"Expected data, received HTML: {tag}/{name}")
                shutil.copyfileobj(response, stream)
            if zipfile.is_zipfile(temp):
                with zipfile.ZipFile(temp) as archive:
                    entries = archive.namelist()
                    if entries == [name] or entries == [Path(name).name]:
                        unpacked = temp.with_suffix(".unpacked")
                        with archive.open(entries[0]) as src, unpacked.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                    else:
                        unpacked = None
                if unpacked is not None:
                    unpacked.replace(temp)
            temp.replace(target)
            records.append({"tag": tag, "file": name})
            print(f"Downloaded {tag}/{name}", flush=True)
    save_json(args.output_dir / "downloaded.json", records)
    print("Run prepare_inputs.py on Compute Node to verify the original complete matrix hashes.")


if __name__ == "__main__":
    main()
