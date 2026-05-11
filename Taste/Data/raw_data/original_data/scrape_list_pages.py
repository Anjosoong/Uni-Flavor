#!/usr/bin/env python3
"""
爬取PMTD主页面列表信息 (共77页)
"""

import requests
import time
import json
import pandas as pd
from bs4 import BeautifulSoup

def scrape_list_page(page_num, max_retries=3):
    """爬取单个列表页面"""
    url = f"https://plantmoleculartastedb.org/PMTDB_browsePhytocompAll.php?page={page_num}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                molecules = []
                
                table = soup.find('table', class_='browsePhytocompounds')
                if table:
                    rows = table.find_all('tr')[1:]
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            form = cells[0].find('form')
                            if form:
                                id_input = form.find('input', {'name': 'id'})
                                if id_input:
                                    molecules.append({
                                        'id': id_input.get('value'),
                                        'name': cells[1].get_text(strip=True),
                                        'taste': cells[2].get_text(strip=True)
                                    })
                return molecules
            else:
                print(f"  ⚠️ 尝试 {attempt + 1}/{max_retries}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ⚠️ 尝试 {attempt + 1}/{max_retries}: {str(e)[:50]}")
            time.sleep(3)
    return None

def scrape_all_list_pages():
    """爬取全部77页列表"""
    total_pages = 77
    all_molecules = []
    
    print(f"开始爬取 {total_pages} 页列表数据...")
    
    for page in range(1, total_pages + 1):
        print(f"[{page}/{total_pages}] 爬取第 {page} 页...")
        
        molecules = scrape_list_page(page)
        if molecules:
            all_molecules.extend(molecules)
            print(f"  ✅ 获取 {len(molecules)} 个分子")
        else:
            print(f"  ❌ 第 {page} 页爬取失败")
        
        time.sleep(2)
    
    # 保存结果
    if all_molecules:
        df = pd.DataFrame(all_molecules)
        df.to_csv('pmtd_molecule_list_all.csv', index=False, encoding='utf-8-sig')
        
        with open('pmtd_molecule_list_all.json', 'w', encoding='utf-8') as f:
            json.dump(all_molecules, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 完成! 共获取 {len(all_molecules)} 个分子")
        print(f"保存到: pmtd_molecule_list_all.csv, pmtd_molecule_list_all.json")
    
    return all_molecules

if __name__ == "__main__":
    scrape_all_list_pages()