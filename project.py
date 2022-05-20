# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:40:08 2022

@author: yakupcatalkaya
"""


import requests
import sys
import time
import html


def get_article_a_year(year=2000,qu=None):
    links = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
    }
    
    year = str(year)
    try:
        for month in range(1,13):
            month = str(month)
            if len(month)==1: month = "0" + month
            for day in range(1,32):
                day = str(day)
                if len(day)==1: day = "0" + day
                link = "http://www.nytimes.com/sitemap/" + year + "/" + month + "/" + day + "/"
                response = requests.post(link, headers=headers)
                if response.status_code!=200:
                    continue
                print(day,"/",month,"/",year)
                cont = response.content.decode("utf-8")
                cont = cont.split("<a href=")
                for con in cont:
                    if "http" in con and "https://www.nytimes.com/" + year + "/" + month + "/" + day in con: 
                        link = con.split('"')[1]
                        if link == "en":
                            continue
                        links.append(link)
        all_link = "\n".join(links)
        if qu !=None:
                    qu.put(all_link)
        return all_link
    except Exception as E:
            print(E)
            
def main_2():        

    try:
        start_time = time.time()
        index = 0     
        for year in range(2000,2023):
            respp = get_article_a_year(year)
            file = open (str(index) + "_link.txt", "w", encoding="utf-8")
            index += 1
            file.write(respp)
            file.close()

        print("The program lasts in ",int(time.time()-start_time)," seconds.")
        
    except Exception as E:
        print(E)
        sys.exit(0)
            
def main_3():
    indexx = 0
    politics, arts, sports, business, technology = 0,0,0,0,0
    file2 = open ("links.txt", "w", encoding="utf-8")
    for year in range(2000,2023):
        file = open (str(indexx) + "_link.txt", "r", encoding="utf-8")
        indexx += 1
        respp = file.read().split("\n")
        for index,line in enumerate(respp):
            if "/technology/" in line: technology+=1
            if"/sports/" in line: sports+=1
            if"/arts/" in line: arts+=1
            if"/business/" in line: business+=1
            if"/politics/" in line: politics+=1
                
            if "/technology/" in line or "/sports/" in line or "/arts/" in line or \
                "/business/" in line or "/politics/" in line:
                    respp[index] = line.replace("https","http")
                    file2.write(respp[index]+"\n")

            
        file.close()
    file2.close()
    print("technology: ",technology)
    print("sports: ",sports)
    print("arts: ",arts)
    print("business: ",business)
    print("politics: ",politics)
    print("all: ",politics+arts+sports+business+technology)
    
def main():
    start_time = time.time()
    link=""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
    }
    try:
        file1 = open("technology.txt", "w", encoding="utf-8")
        file2 = open("sports.txt", "w", encoding="utf-8")
        file3 = open("arts.txt", "w", encoding="utf-8")
        file4 = open("business.txt", "w", encoding="utf-8")
        file5 = open("politics.txt", "w", encoding="utf-8")
        
        prob = open("problems.txt", "w", encoding="utf-8")
        
        file = open ("links.txt", "r", encoding="utf-8").read().split("\n")
        number = len(file)
        politics, arts, sports, business, technology = 0,0,0,0,0
        indexxx = 0
        for link in file:
            indexxx += 1
            if indexxx % 1000==0:print(int((time.time()-start_time)*(number/indexxx))," seconds "," % ",indexxx/number, indexxx,". ", " link: ",link)
            response = requests.post(link, headers=headers)
            decresponse = html.unescape(response.content.decode("latin"))
            cont = decresponse.split("<body>")[1].split("<p class")[3:]
            metin = []
            for con in cont:
                line = con.split(">")[1].split("<")[0]
                metin.append(line)
            xx = " ".join(metin) + "\n" + "*"*50 + "\n"
            if "/technology/" in link: 
                technology+=1
                file1.write(xx)
            if"/sports/" in link: 
                sports+=1
                file2.write(xx)
            if"/arts/" in link: 
                arts+=1
                file3.write(xx)
            if"/business/" in link: 
                business+=1
                file4.write(xx)
            if"/politics/" in link: 
                politics+=1
                file5.write(xx)
    except Exception as E:
        print(E)
        print(link)
        prob.write(E + "\n" + link + "\n")
    print("technology: ",technology)
    print("sports: ",sports)
    print("arts: ",arts)
    print("business: ",business)
    print("politics: ",politics)
    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
    file.close()
    prob.close()
    
    print("The program lasts in ",int(time.time()-start_time)," seconds.")
    
if __name__=="__main__":
    main()
    