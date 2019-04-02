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
import io
import bz2
import re
import psutil
process = psutil.Process(os.getpid())
import shutil
from  tqdm import tqdm

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
files_to_process = sorted([f for f in os.listdir(dataset_directory) if os.path.isfile(os.path.join(dataset_directory, f)) and f not in processed_files])


if cfg.getboolean("core", "ramfs_use"):
    print("USING RAMFS")
    working_dir = "wikidata_parser"
    ramfs_dir = cfg.get("core", "ramfs_dir")
    
    ramfs_dir = os.path.join(ramfs_dir, working_dir)
    if not os.path.isdir(ramfs_dir):
        os.mkdir(ramfs_dir)
        
    ramfs_files = os.listdir(ramfs_dir)
    
    for f in files_to_process:
        if f in ramfs_files:
            continue
        else:
            print("Copying file: {f}".format(f=f))
            shutil.copy2(os.path.join(dataset_directory, f), os.path.join(ramfs_dir, f))
            print("Done...")
    
    dataset_files = sorted([os.path.join(ramfs_dir, f) for f in files_to_process])
else:
    dataset_files = sorted([os.path.join(dataset_directory, f) for f in files_to_process])


print(dataset_files)

class States:
    IGNORE = -1
    DEFAULT = 0
    SITEINFO = 1
    PAGE = 2


def parse_page_iter(page_content, dataset_file):
    ts_min = pd.to_datetime(cfg.get("preprocessing", "timestamp_from"), utc=True)
    ts_max = pd.to_datetime(cfg.get("preprocessing", "timestamp_to"), utc=True)
    revisions = []
    
    try:
        content_buffer = io.BytesIO(page_content.encode("utf-8"))
        xml_context = etree.iterparse(content_buffer, tag=("revision", "title", "ns", "id"), huge_tree=True)
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
    
    title = "" #xml_content.find("title").text
    ns = "" #xml_content.find("ns").text
    page_id = "" #xml_content.find("id").text
    
    for action, elem in xml_context:
                
        if action != "end":
            continue
        if elem.getparent().tag == "page":
            if elem.tag == "title":
                title = elem.text
            elif elem.tag == "ns":
                ns = elem.text
            elif elem.tag == "id":
                page_if = elem.text
            continue
          
        if elem.tag != "revision":
            continue # we probably found the id field of a revision here. we grab that as a whole for the revision element
            
        # now we now elem is a revision
        revision = elem

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
                if revision.find("text") is None:
                    note = ""
                else:
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
    
    
def parse_page(page_content, dataset_file):
    ts_min = pd.to_datetime(cfg.get("preprocessing", "timestamp_from"), utc=True)
    ts_max = pd.to_datetime(cfg.get("preprocessing", "timestamp_to"), utc=True)
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
        raise
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
                if revision.find("text") is None:
                    note = ""
                else:
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
    try:
        curr_proc = current_process()._identity[0]
    except Exception as e:  # we have no current_process which means it is probably started not as a multiprocess thing
        curr_proc = 1
    split_parsing = cfg.getboolean("core", "split_parsing")
    split_limit = cfg.getint("core", "split_limit")
    split_index = 0
    page_counter = 0
    processor_index = curr_proc % cfg.getint("core", "num_cores")
    bar_offset = processor_index
    revisions = []
    filesize = os.path.getsize(dataset_file)/(1024**2)
    open_func = None
    if dataset_file.endswith("bz2"):
        open_func = bz2.open
    else:
        open_func = open
    with open_func(dataset_file, "rb") as data_file, tqdm(position=(bar_offset-1), desc=os.path.basename(dataset_file)) as progress_bar:
        carry_content = ""
        state = States.DEFAULT
        text_state = States.DEFAULT
        for i,row in enumerate(data_file):
            tmp_idx = "S: {s}, Pages: {p}, Line {l}, Len: {len}".format(s=state,p=page_counter, l=i, len=len(row))
            if row.startswith(b"      <text"):
                text_state = States.IGNORE
            elif row.endswith(b"</sha1>\n"):
                text_state = States.DEFAULT
            
            if text_state == States.IGNORE:
                continue
            
            row_decoded = row.decode("utf-8")

            
            if state == States.DEFAULT:
                if row == b'  <page>\n':
                    carry_content += row_decoded
                    state = States.PAGE
            
            elif state == States.PAGE:
                carry_content += row_decoded

                if row == b'  </page>\n':
                    state = States.DEFAULT
                    if cfg.getboolean("core", "iter_parsing"):
                        parsed_page = parse_page_iter(carry_content, dataset_file)
                    else:
                        parsed_page = parse_page(carry_content, dataset_file)
                    if parsed_page is not None:
                        revisions += parsed_page
                        page_counter +=1
                        progress_bar.update()
                    carry_content = ""
            if split_parsing:
                if page_counter == split_limit:
                    
                    ts_now_str = str(pd.datetime.now())[:-7].replace(" ","_")
                    dump_filename = os.path.join(cfg.get("directory", "pickles_split"), "df_revisions-[{file}]{ts_now}[{r_from}-{r_to}].p".format(file=os.path.basename(dataset_file), 
                                                                                                                                                  ts_now=ts_now_str, 
                                                                                                                                                  r_from=split_index*split_limit+1, 
                                                                                                                                                  r_to=(split_index+1)*split_limit
                                                                                                                                                 )
                                                )
                    df_revisions = pd.DataFrame(revisions)
                    df_revisions.to_pickle(dump_filename)
                    print("Writing split file: {fn}".format(fn=dump_filename))
                    split_index +=1
                    revisions = []
                    page_counter = 0
    
    ts_now_str = str(pd.datetime.now())[:-7].replace(" ","_")
    print("parsing complete")
    if not split_parsing:
        dump_filename = os.path.join(storage_directory, "df_revisions-[{file}]{ts_now}.p".format(file=os.path.basename(dataset_file), ts_now=ts_now_str))
        df_revisions = pd.DataFrame(revisions)
        df_revisions.to_pickle(dump_filename)
        print("Writing file: {fn}".format(fn=dump_filename))
    
    return True    

#for i,f in enumerate(dataset_files):
#    parse_file(f)

with Pool(cfg.getint("core", "num_cores"), maxtasksperchild=1) as processor_pool:
    success = list(processor_pool.imap(parse_file, dataset_files))
