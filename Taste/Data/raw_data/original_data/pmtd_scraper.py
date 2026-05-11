#!/usr/bin/env python3
"""
PMTD网站爬虫 - 简单直接的数据爬取脚本
"""

import requests
import time
import json
import pandas as pd
from bs4 import BeautifulSoup
import re
import os

def parse_molecule_page(html_content, molecule_id):
    """解析分子页面数据 - 获取所有信息"""
    soup = BeautifulSoup(html_content, 'html.parser')
    data = {'id': molecule_id}
    
    # 获取标题
    title = soup.find('h1')
    if title:
        data['name'] = title.get_text(strip=True)
    
    # 解析所有数据字段 - 保留完整信息包括引用
    title_divs = soup.find_all('div', id='title')
    for title_div in title_divs:
        title_text = title_div.get_text(strip=True)
        content_div = title_div.find_next_sibling('div', id='content')
        
        if content_div:
            # 保留完整的HTML内容，包括引用链接
            content_text = content_div.get_text(strip=True)
            content_html = str(content_div)
            
            if 'Molecular formula' in title_text:
                data['molecular_formula'] = content_text
                
            elif 'Taste and/or trigeminal orosensation' in title_text:
                # 这是真正的味道信息，保留完整内容包括引用
                data['taste_orosensation'] = content_text
                data['taste_orosensation_html'] = content_html
                
            elif 'Sensory data' in title_text:
                data['sensory_data'] = content_text
                data['sensory_data_html'] = content_html
                
            elif 'Taste receptors' in title_text:
                data['taste_receptors'] = content_text
                data['taste_receptors_html'] = content_html
                
            elif 'Chemical class' in title_text:
                data['chemical_class'] = content_text
                
            elif 'Synonyms' in title_text:
                data['synonyms'] = content_text
                
            elif 'antiinflammatory' in title_text:
                data['antiinflammatory'] = content_text
                data['antiinflammatory_html'] = content_html
                
            elif 'References' in title_text:
                data['references'] = content_text
                data['references_html'] = content_html
                
            elif 'Identifiers' in title_text:
                data['identifiers_raw'] = content_text
                # 解析数据库ID
                patterns = {
                    'pubchem_cid': r'PubChem CID:\s*(\d+)',
                    'foodb_id': r'FooDB ID:\s*([A-Z0-9]+)',
                    'hmdb_id': r'HMDB ID:\s*([A-Z0-9]+)',
                    'chembl_id': r'ChEMBL ID:\s*([A-Z0-9]+)',
                    'bitterdb_id': r'BitterDB ID:\s*(\d+)'
                }
                for key, pattern in patterns.items():
                    match = re.search(pattern, content_text)
                    if match:
                        data[key] = match.group(1)
    
    # 获取SMILES
    canonical_input = soup.find('input', id='canSMILES_copy')
    if canonical_input:
        data['canonical_smiles'] = canonical_input.get('value', '')
    
    isomeric_input = soup.find('input', id='isoSMILES_copy')
    if isomeric_input:
        data['isomeric_smiles'] = isomeric_input.get('value', '')
    
    # 获取页面的完整HTML用于备份
    data['full_html'] = html_content
    
    return data

def scrape_molecule(molecule_id, max_retries=3):
    """爬取单个分子数据 - 支持重试机制"""
    url = f"https://plantmoleculartastedb.org/compound.php?id={molecule_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)  # 增加超时时间
            if response.status_code == 200:
                return parse_molecule_page(response.text, molecule_id)
            else:
                print(f"  ⚠️ 尝试 {attempt + 1}/{max_retries}: HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"  ⏱️ 尝试 {attempt + 1}/{max_retries}: 超时，等待重试...")
            time.sleep(5)
        except Exception as e:
            print(f"  ⚠️ 尝试 {attempt + 1}/{max_retries}: {str(e)[:50]}")
            time.sleep(3)
    
    print(f"  ❌ {molecule_id}: 重试{max_retries}次后失败")
    return None

def scrape_all_molecules():
    """爬取所有分子数据 - 完整版本，支持VPN不稳定"""
    
    # 读取分子列表或生成ID范围
    if os.path.exists("pmtd_molecule_list.csv"):
        df = pd.read_csv("pmtd_molecule_list.csv")
        molecule_ids = df['id'].tolist()
    elif os.path.exists("../pmtd_molecule_list.csv"):
        df = pd.read_csv("../pmtd_molecule_list.csv")
        molecule_ids = df['id'].tolist()
    else:
        # 生成全部ID范围 (PMTDB00001 到 PMTDB02000，覆盖网站全部数据)
        molecule_ids = [f"PMTDB{i:05d}" for i in range(1, 2001)]
    
    print(f"开始爬取 {len(molecule_ids)} 个分子的完整数据...")
    print("包括: 味道信息、感官数据、受体信息、引用等所有内容")
    print("网络不稳定时会自动重试(最多3次)\n")
    
    all_data = []
    success_count = 0
    failed_ids = []
    
    for i, molecule_id in enumerate(molecule_ids, 1):
        print(f"\n[{i}/{len(molecule_ids)}] 爬取 {molecule_id}...")
        
        data = scrape_molecule(molecule_id)
        if data:
            all_data.append(data)
            success_count += 1
            print(f"  ✅ 成功 - 获取了 {len(data)} 个字段")
            
            # 显示关键信息
            if 'taste_orosensation' in data:
                taste_preview = data['taste_orosensation'][:80] + "..." if len(data.get('taste_orosensation', '')) > 80 else data.get('taste_orosensation', '')
                print(f"     味道: {taste_preview}")
        else:
            failed_ids.append(molecule_id)
        
        # 延迟避免被封
        time.sleep(2)
        
        # 每10个分子保存一次，防止数据丢失
        if i % 10 == 0:
            print(f"\n  💾 中间保存 (已处理 {i} 个分子, 成功 {success_count} 个)...")
            with open('pmtd_all_molecules_backup.json', 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    # 保存最终结果
    if all_data:
        # 完整JSON格式 (包含HTML)
        with open('pmtd_all_molecules_complete.json', 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        # 简化版本用于CSV (移除HTML字段)
        simplified_data = []
        for item in all_data:
            simplified_item = {k: v for k, v in item.items() if not k.endswith('_html') and k != 'full_html'}
            simplified_data.append(simplified_item)
        
        df_result = pd.DataFrame(simplified_data)
        df_result.to_csv('pmtd_all_molecules_complete.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n🎉 爬取完成!")
        print(f"成功: {success_count}/{len(molecule_ids)}")
        print(f"完整数据 (含HTML): pmtd_all_molecules_complete.json")
        print(f"简化数据 (CSV): pmtd_all_molecules_complete.csv")
        
        if failed_ids:
            print(f"\n⚠️ 失败的分子ID: {failed_ids}")
            with open('failed_molecules.txt', 'w') as f:
                f.write('\n'.join(failed_ids))
            print(f"失败列表已保存到: failed_molecules.txt")
    
    return all_data

if __name__ == "__main__":
    scrape_all_molecules()