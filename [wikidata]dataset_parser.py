################################################################################################################################################################
# Author: Patrick Kasper
# Dataset Source: https://dumps.wikimedia.org/wikidatawiki/20190101/
# Note: All files under: 2019-01-16 12:41:51 done All pages with complete page edit history (.bz2)
################################################################################################################################################################

import bz2
import json

from tqdm import tqdm

import numpy as np
import pandas as pd

import os
import bz2
import re
import psutil
process = psutil.Process(os.getpid())
import shutil

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import current_process, Pool

from xml.etree import ElementTree
from lxml import etree

from configparser import ConfigParser
cfg = ConfigParser()
cfg.read("config.cfg")

dataset_directory = os.path.join(cfg.get("directory", "dataset"), cfg.get("dataset", "wikidata"))
storage_directory = cfg.get("directory", "pickles")

processed_files = [re.search(r"\[(.*)\]", x)[1] for x in os.listdir(storage_directory) if os.path.isfile(os.path.join(storage_directory, x))] 
dataset_files = sorted([os.path.join(dataset_directory, f) for f in os.listdir(dataset_directory) if os.path.isfile(os.path.join(dataset_directory, f)) and f not in processed_files])

print(dataset_files)

class States:
    DEFAULT = 0
    SITEINFO = 1
    PAGE = 2

def parse_page(page_content, dataset_file):
    ts_min = pd.to_datetime("2017-01-01T00:00:00Z", utc=True)
    ts_max = pd.to_datetime("2018-12-31T23:59:59Z", utc=True)
    revisions = []
    try:
        xml_parser = etree.XMLParser(huge_tree=True)
        xml_content = etree.fromstring(page_content, parser=xml_parser)
    except Exception as e:
        print(dataset_file)
        print(len(page_content))
        ts_now_str = str(pd.datetime.now())[:-7].replace(" ","_")
        dump_filename = "error_dump-[{file}]{ts_now}.txt".format(file=os.path.basename(dataset_file), ts_now=ts_now_str)
        dump_filename = os.path.join(cfg.get("directory", "error_dumps"), dump_filename)
        print("PRE Dump filename: {fn}".format(fn=dump_filename))
        with open(dump_filename, "w") as error_dump:
            error_dump.write(page_content)
        print("POST Dump filename: {fn}".format(fn=dump_filename))
        
        
        print(e)
        return None
        
    title = xml_content.find("title").text
    ns = xml_content.find("ns").text
    page_id = xml_content.find("id").text
    
    for revision in xml_content.findall("revision"):
        try:
            ts_raw = revision.find("timestamp").text
            timestamp = pd.to_datetime(ts_raw, utc=True)
            if not (ts_min <= timestamp <= ts_max):
                continue
            user = revision.find("contributor")
            if "deleted" in user.attrib:
                continue
            if user.find("username") is None:
                user_name = ""
                user_id = np.NaN
                user_ip = user.find("ip").text
            else:                
                user_name = user.find("username").text
                user_id = user.find("id").text
                user_ip = ""
                
            if revision.find("comment") is None:
                note = revision.find("text").text
                comment = ""
            else:
                note = ""
                comment = revision.find("comment").text
                                
            model = revision.find("model").text
        except Exception as e:
            print(dataset_file)
            print(e)
            print(etree.dump(revision))
            raise
        
        revision_data = {
            "page_title": title,
            "page_ns": ns,
            "page_id": page_id,
            "timestamp": timestamp,
            "user_name": user_name,
            "user_id": user_id,
            "user_ip": user_ip,
            "comment": comment,
            "model": model,
            "note": note,
            "dataset_file": os.path.basename(dataset_file)
        }
        revisions.append(revision_data)
    return revisions
    
    
def parse_file(dataset_file):   
    processor_index = current_process()._identity[0] % cfg.getint("core", "num_cores")
    bar_offset = processor_index
    revisions = []
    filesize = os.path.getsize(dataset_file)/(1024**2)
    with bz2.open(dataset_file, "rb") as data_file:
        carry_content = ""
        state = States.DEFAULT
        for row in data_file:
            row_decoded = row.decode("utf-8")
            
            if state == States.DEFAULT:
                if row == b'  <page>\n':
                    carry_content += row_decoded
                    state = States.PAGE
            
            elif state == States.PAGE:
                carry_content += row_decoded
                if row == b'  </page>\n':
                    state = States.DEFAULT
                    parsed_page = parse_page(carry_content, dataset_file)
                    if parsed_page is not None:
                        revisions += parsed_page
                    carry_content = ""
    
    ts_now_str = str(pd.datetime.now())[:-7].replace(" ","_")
    dump_filename = os.path.join(storage_directory, "df_revisions-[{file}]{ts_now}.p".format(file=os.path.basename(dataset_file), ts_now=ts_now_str))
    df_revisions = pd.DataFrame(revisions)
    df_revisions.to_pickle(dump_filename)
    print("Writing file: {fn}".format(fn=dump_filename))
    
    return True    

with Pool(cfg.getint("core", "num_cores"), maxtasksperchild=1) as processor_pool:
    success = list(processor_pool.imap(parse_file, dataset_files))
