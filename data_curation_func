import re
import time
import json
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import os
import random
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import sys
from requests import get 
import pickle
import datetime
import gc


# PARSE GENE FILES TO GET EXON BOUNDARIES
def numeric(string):
    return int(re.sub("[^0-9]", "", string))

def parse_gene_files(input_file,output_file):

    f = open(input_file,'r')
    output=output_file


    aux=[]
    data=[]
    cds=False
    name=0
    total_bs_len=0 
    reading_frame=1 # assuming that if it isn't specified, start translation from the first nucleotide
    partial_5=False
    partial_3=False
    while True:
        l = f.readline()
        if len(l)>0:


            if l.startswith('ID'):
                name=l.split(';')[0].split(' ')[-1]
            if l.startswith('FT   CDS'): 
                cds=True
                
                            
            if l.startswith('FT                   /codon_start'): #CHANGE: save reading frame
                reading_frame=int(l.split('=')[-1])
               # print(reading_frame)

                
            
            if len(aux)>0:
                if l.startswith('FT                   /'):
                    cds=False
                #last line to appear 
                if l.startswith('SQ   Sequence'): #CHANGE: save total lenght to compare
                    total_bs_len=int(l.split(';')[0].split(' ')[-2])
                    
                    exons=[a.split('..') for a in aux]
                    #print(exons)
                    if len(exons)>1: 
                        exon_len=[]
                        for i,exon in enumerate(exons):
                            if len(exon)==2:
                                    exon_len.append(numeric(exon[1])-numeric(exon[0])+1) #CHANGE: +1 (real LENGTH)
                                    
                        if len(exon_len)>0: #just double checking
                                                        
                            len_dif=sum(exon_len)-total_bs_len
                            
                            phases_ok=True
                            
                            exon_len[0]+=-(reading_frame-1)
                            #exon_bs_real=np.cumsum(exon_len)[:-1] 
                            exon_bs_inAA=(np.cumsum(exon_len)[:-1])//3
                            intron_phases_=np.array(exon_len)%3 
                            intron_phases=intron_phases_[:-1]
                            if sum(intron_phases_)%3!=0:
                                phases_ok=False
                                #print('error in total phase',reading_frame)
                            exon_sym=(np.hstack([np.array([0]),intron_phases])-intron_phases_)==0
                            junction_both_sym=exon_sym[1:]*exon_sym[:-1]
                            last_exon_sym=exon_sym[:-1]
                            next_exon_sym=exon_sym[1:]
                            exon_len[0]+=(reading_frame-1) # recover this as it was originally
                            data.append([name,exon_len,exon_bs_inAA,total_bs_len,
                                         reading_frame,partial_5,partial_3,len_dif,
                                         exon_sym,junction_both_sym,last_exon_sym,next_exon_sym,
                                         intron_phases,phases_ok])
                    aux=[]
                    name=0
                    partial_5=False
                    partial_3=False
                    total_bs_len=0
                    reading_frame=1

                #print(l)

            if cds:
                if '>' in l: #CHANGE: save if 3’ partial CDS
                    partial_3=True
                if '<' in l: #CHANGE: save if 5’ partial CDS
                    partial_5=True
                a=[re.findall(r'\b:\S+',i) for i in l.split(',')]
                for a_ in a:
                    if len(a_)>0:
                        aux.append(a_[0][1:])

            

        else:
            with open(output, 'wb') as fp:
                pickle.dump(data,fp)
            break

    f.close()


def load_exons(output_file,code_table):
        with open (output_file, 'rb') as fp:
              all_exon_len = pickle.load(fp)

        exon_b=pd.DataFrame(all_exon_len)
        exon_b.columns=['gene_bank_id','exon_len','exon_bs_inAA','total_bs_len',
                        'reading_frame','partial_5','partial_3','len_dif','exon_sym','junction_both_sym',
                        'last_exon_sym','next_exon_sym',
                        'intron_phases','phases_ok']
        code_table['gene_bank_id']=[x.split('.')[0] for x in code_table.to]
        exon_b_full=exon_b.merge(code_table)
        exon_b_full.columns=['gene_bank_id','exon_len','exon_bs_inAA','total_bs_len','reading_frame',
                             'partial_5','partial_3','len_dif','exon_sym','junction_both_sym',
                            'last_exon_sym','next_exon_sym',
                             'intron_phases','phases_ok','uniprot_id','gene_bank_id_long']
        return exon_b_full
    
def translate_exon_pos(exon_b_full,MSA,col_nogap,names_table):
    all_bs_aa=[]
    ins_bs=[]
    intron_ph=[]
    last_ex_ph=[]
    next_ex_ph=[]
    both_ex_ph=[]
    
    for i,exs in enumerate(exon_b_full.exon_bs_inAA):

        seq_ix=np.arange(len(names_table))[names_table.name==exon_b_full.name[i]][0]
        ali_seq_num=np.zeros(MSA.shape[1],dtype=int)-1
        ali_seq_num[MSA[seq_ix,:]!='-']=np.arange(sum(MSA[seq_ix,:]!='-'))+names_table.pos_ini[seq_ix]
        ali_seq_num=ali_seq_num[col_nogap]
        bs_aa=[]
        ins_bs_i=[]
        ali_ix=[]
        
        for iex,ex in enumerate(exs):

            if ex>=exon_b_full.pos_ini[i]:
                if  ex<=exon_b_full.pos_fin[i]:
                    aux=True
                    insertion=False
                    while aux:
                        a=np.where(ali_seq_num==ex)[0]
                        #a=np.where(MnumA_nogap[names_table.name==exon_b_full.name[i],:][0]==ex)[0]
                        ex=ex-1
                        if len(a)>0:
                            bs_aa.append(a[0])
                            ins_bs_i.append(insertion)
                            ali_ix.append(iex)
                            aux=False
                        elif (ex+1)<ali_seq_num[0]: 
                            insertion=True
                            bs_aa.append(0)
                            ins_bs_i.append(insertion)
                            ali_ix.append(iex)
                            aux=False
                      #  if ex==exon_b_full.pos_ini[i]: # esto no esta bien. duplica estos casos! no hace falta
                      #      bs_aa.append(a[0])
                      #      ins_bs_i.append(insertion)
                      #      aux=False
                        insertion=True
                        
        intron_ph.append(exon_b_full.intron_phases[i][ali_ix])
        last_ex_ph.append(exon_b_full.last_exon_sym[i][ali_ix])
        next_ex_ph.append(exon_b_full.next_exon_sym[i][ali_ix])
        both_ex_ph.append(exon_b_full.junction_both_sym[i][ali_ix])
        all_bs_aa.append(bs_aa)
        ins_bs.append(ins_bs_i)
    return all_bs_aa, ins_bs, intron_ph, last_ex_ph, next_ex_ph, both_ex_ph
    


    
def family_ali_exon(family_code,interpro_code,family_name,path,pdb_seq,pdb_uni_beg,
                    cdhit_sim=0.9,DOWNL=True,RUN_CDHIT=True,PARSE_CDHIT=True):
    if DOWNL:
        DATE=str(datetime.date.today().day)+'/'+str(datetime.date.today().month)+'/'+str(datetime.date.today().year)
    else:
        DATE='no download'
    if not RUN_CDHIT:
        cdhit_sim=np.nan
        
    ##### DOWNLOAD DATA  #####   
    
    def download(url, file_name):
        with open(file_name, "wb") as file:
            response = get(url)
            file.write(response.content)
        
    # DOWNLOAD PFAM ALIGNMENT FROM INTERPRO      
    stname=family_code+'_uniprot_st.txt'
    fastaname=family_code+'_uniprot.txt'   
    Ali_url='https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/'+family_code+'/?annotation=alignment:uniprot'
 
    if DOWNL:
        download(Ali_url, path+stname)
    
        with open(path+stname,'r') as inFile: # stockholm to fasta format
            with open(path+fastaname, "w") as output_handle:
                SeqIO.write(list(SeqIO.parse(inFile,'stockholm')), output_handle, "fasta")
        os.system('rm '+path+stname)

    # MAKE MSA AND UNIPROT ID (name) LIST FROM FASTA
    fasta_sequences = list(SeqIO.parse(path+fastaname,'fasta'))
    MSA=np.empty([len(fasta_sequences),len(fasta_sequences[0].seq)],dtype='<U1')
    names=[]
    for j in range(len(fasta_sequences)):
        MSA[j,:]=[i for i in fasta_sequences[j].seq]
        names.append(fasta_sequences[j].id)
    name_list=[names[i].split('.')[0] for i in range(len(names))]
    ini=[int(names[i].split('/')[1].split('-')[0]) for i in range(len(names))]
    fin=[int(names[i].split('/')[1].split('-')[1]) for i in range(len(names))]

    names_table=pd.DataFrame({'name':names,'uniprot_id':name_list, 'pos_ini':ini,'pos_fin':fin})
    names_table['msa_index']=names_table.index

    # DOWNLOAD TAX ID LINEAGE FOR EACH PROTEIN FROM UNIPROT
    tax_info_url='https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Clineage%2Clineage_ids&format=tsv&query=%28'+interpro_code+'%29'
    
    if DOWNL:
        download(tax_info_url, path+'tax_id')

    # FILTER: KEEP ONLY NON-BACTERIA ORGANISMS
    tid=pd.read_csv(path+'tax_id',sep='\t')
    tid['Bacteria']=['Bacteria' in tid['Taxonomic lineage'][i] for i in range(len(tid))]
    names_table=names_table.merge(tid,left_on='uniprot_id',right_on='Entry')
    #nonbac_name_list=names_table.uniprot_id[~names_table.Bacteria].to_list()

    # MEMORY CLEAN
    del tid
    gc.collect()

    
    # GET GENE CODES (GENBANK) USING UNIPROT MAP ID API 
    POLLING_INTERVAL = 3
    API_URL = "https://rest.uniprot.org"
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))

    def check_response(response):
        try:
            response.raise_for_status()
        except requests.HTTPError:
            print(response.json())
            raise


    def submit_id_mapping(from_db, to_db, ids):
        request = requests.post(
            f"{API_URL}/idmapping/run",
            data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
        )
        check_response(request)
        return request.json()["jobId"]


    def get_next_link(headers):
        re_next_link = re.compile(r'<(.+)>; rel="next"')
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)


    def check_id_mapping_results_ready(job_id):
        while True:
            request = session.get(f"{API_URL}/idmapping/status/{job_id}")
            check_response(request)
            j = request.json()
            if "jobStatus" in j:
                if j["jobStatus"] == "RUNNING":
                   # print(j)
                    print(f"Retrying in {POLLING_INTERVAL}s")
                    time.sleep(POLLING_INTERVAL)
                else:
                    raise Exception(j["jobStatus"])
            else:
                print(j)
                return bool(j["results"] or j["failedIds"])
                

    def get_batch(batch_response, file_format, compressed):
        batch_url = get_next_link(batch_response.headers)
        while batch_url:
            batch_response = session.get(batch_url)
            batch_response.raise_for_status()
            yield decode_results(batch_response, file_format, compressed)
            batch_url = get_next_link(batch_response.headers)


    def combine_batches(all_results, batch_results, file_format):
        if file_format == "json":
            for key in ("results", "failedIds"):
                if key in batch_results and batch_results[key]:
                    all_results[key] += batch_results[key]
        elif file_format == "tsv":
            return all_results + batch_results[1:]
        else:
            return all_results + batch_results
        return all_results


    def get_id_mapping_results_link(job_id):
        url = f"{API_URL}/idmapping/details/{job_id}"
        request = session.get(url)
        check_response(request)
        return request.json()["redirectURL"]


    def decode_results(response, file_format, compressed):
        if compressed:
            decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
            if file_format == "json":
                j = json.loads(decompressed.decode("utf-8"))
                return j
            elif file_format == "tsv":
                return [line for line in decompressed.decode("utf-8").split("\n") if line]
            elif file_format == "xlsx":
                return [decompressed]
            elif file_format == "xml":
                return [decompressed.decode("utf-8")]
            else:
                return decompressed.decode("utf-8")
        elif file_format == "json":
            return response.json()
        elif file_format == "tsv":
            return [line for line in response.text.split("\n") if line]
        elif file_format == "xlsx":
            return [response.content]
        elif file_format == "xml":
            return [response.text]
        return response.text


    def get_xml_namespace(element):
        m = re.match(r"\{(.*)\}", element.tag)
        return m.groups()[0] if m else ""


    def merge_xml_results(xml_results):
        merged_root = ElementTree.fromstring(xml_results[0])
        for result in xml_results[1:]:
            root = ElementTree.fromstring(result)
            for child in root.findall("{http://uniprot.org/uniprot}entry"):
                merged_root.insert(-1, child)
        ElementTree.register_namespace("", get_xml_namespace(merged_root[0]))
        return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)


    def print_progress_batches(batch_index, size, total):
        n_fetched = min((batch_index + 1) * size, total)
       # print(f"Fetched: {n_fetched} / {total}")


    def get_id_mapping_results_search(url):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        file_format = query["format"][0] if "format" in query else "json"
        if "size" in query:
            size = int(query["size"][0])
        else:
            size = 500
            query["size"] = size
        compressed = (
            query["compressed"][0].lower() == "true" if "compressed" in query else False
        )
        parsed = parsed._replace(query=urlencode(query, doseq=True))
        url = parsed.geturl()
        request = session.get(url)
        check_response(request)
        results = decode_results(request, file_format, compressed)
        total = int(request.headers["x-total-results"])
        if total>0:
          #  print_progress_batches(0, size, total)
            for i, batch in enumerate(get_batch(request, file_format, compressed), 1):
                results = combine_batches(results, batch, file_format)
           #     print_progress_batches(i, size, total)
            if file_format == "xml":
                return merge_xml_results(results)
        else:
            results=[]
        return results


    def get_id_mapping_results_stream(url):
        if "/stream/" not in url:
            url = url.replace("/results/", "/results/stream/")
        request = session.get(url)
        check_response(request)
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        file_format = query["format"][0] if "format" in query else "json"
        compressed = (
            query["compressed"][0].lower() == "true" if "compressed" in query else False
        )
        return decode_results(request, file_format, compressed)
    
    res_dict=[]
    if DOWNL:
        name_array_=names_table.uniprot_id[~names_table.Bacteria].unique()
        print(len(name_array_))
        if len(name_array_)>10000:
            name_array_sep=np.array_split(name_array_, len(name_array_)//10000)  
        else:
            name_array_sep=[name_array_]
        results=[]
        for j,array_j in enumerate(name_array_sep):
            if len(array_j)>0:
                print('len aj=',len(array_j))
                print(array_j)
                job_id = submit_id_mapping(
                    from_db="UniProtKB_AC-ID", to_db="EMBL-GenBank-DDBJ_CDS", ids=list(array_j))
                if check_id_mapping_results_ready(job_id):
                    link = get_id_mapping_results_link(job_id)
                    results = get_id_mapping_results_search(link)
                    # Equivalently using the stream endpoint which is more demanding
                    # on the API and so is less stable:
                    # results = get_id_mapping_results_stream(link)
                #if j==0:
                #    res_dict=results['results']
                #else:
                if len(results)>0:
                    res_dict+=results['results']
        if len(res_dict)==0:
            print('NO BAC')
            code_table=pd.DataFrame(['no_bac'])

        else:
            code_table=pd.DataFrame(res_dict) 
            code_table=code_table[~code_table.duplicated(subset='from')].reset_index(drop=True) # eliminate duplicated genes per protein
        code_table.to_csv(path+'code_table.csv',index_label=False)
    
    
    code_table=pd.read_csv(path+'code_table.csv') #400k
    
    if len(code_table)>2:
        # GET GENE FILES (SLOW) FOR EACH GENE CODE CORRESPONDING TO A NON-BACTERIA ORGANISM
        name_list=['https://www.ebi.ac.uk/ena/browser/api/embl/'+code+'?lineLimit=1000' for code in code_table.to]

        with open(path+'name_list_all.txt', 'w') as f:
            f.write("\n".join(map(str, name_list)))

        if DOWNL:
            os.system('wget -i '+path+'name_list_all.txt'+' -O '+path+'complete_exon_info_all.txt')

         # MEMORY CLEAN
        del name_list
        gc.collect()





        input_file=path+"complete_exon_info_all.txt"
        output_file=path+"exon_len_info_all.txt"

     #   print('parsing gene file')
        parse_gene_files(input_file,output_file)
    #    exon_b_full=load_exons(output_file,code_table)
    else:
        print('no non-bacteria proteins')

    del code_table
    gc.collect()
    
    # DOWNLOAD FULL SEQUENCES 
    
    full_uniprot_url='https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28'+interpro_code+'%29'
    
    unaligned_fasta=path+family_code+'_uniprot_unaligned.fasta'
    if DOWNL:
        download(full_uniprot_url, unaligned_fasta)  
    
    # CLUSTERING - CDHIT 90%
    
    cd_hit_out=path+family_code+'_cd_hit'
    w_file = cd_hit_out+".weights"

    if RUN_CDHIT:
        print('running cdhit')

        os.system('cdhit -c '+str(cdhit_sim)+' -d 0 -i '+unaligned_fasta+' -o '+cd_hit_out)
        
    if PARSE_CDHIT:
        print('parse cdhit file')
        #parse full sequences cd-hit clusters
        f = open(cd_hit_out+".clstr",'r')
        os.system('rm '+w_file)
        name = []
        seq_len=[]
        C = -1 #asi arranca en 0 y mantenemos la notación

        while True:
            l = f.readline()
            if len(l)>0:
                if l[0] == '>': 
                    if name != []:
                        w = [1.0/len(name)] * len(name)
                        data = pd.DataFrame({'name':name,'weight':w,'cluster':C,'seq_len':seq_len})
                        data.to_csv(w_file,sep=',',mode='a',header=False,index=False)
                        name = []
                        seq_len=[]

                    C = C +1
                    if C%10000==0:
                        print(C)

                elif int(l[0]) >= 0:
                    m = re.search('(?<=\|)(.*)(?=\|)',l) #es el nombre de la seq
                    name.append(m.group(0))
                    seq_len.append(int(l.split('aa')[0].split('\t')[-1]))

            else:
                w = [1.0/len(name)] * len(name)
                print('last_cluster',name,w,C)
                data = pd.DataFrame({'name':name,'weight':w,'cluster':C,'seq_len':seq_len})
                data.to_csv(w_file,sep=',',mode='a',header=False,index=False)
                break
        f.close()
        print('end of cdhit file parsing')
        
    cdhit_table=pd.read_csv(w_file,header=None)
    cdhit_table.columns=['uniprot_id','w','cluster','seq_len']    
    
    names_table=names_table.merge(cdhit_table,on='uniprot_id') #ESTA TABLA TIENE TODA LA INFO CONCENTRADA X ENTRADA
    #names_table.to_csv(path+'info_alignment.csv',index_label=False)
    with open(path+'info_alignment', 'wb') as fp:
        pickle.dump(names_table,fp)
    
    del cdhit_table
    gc.collect()
    
    
    # REDEFINE MSA 

    MSA=MSA[names_table.msa_index,:] # uniprot depreciated codes out
   # print('clean MSA')

    # REMOVE COLUMNS WITH MORE THAN 70% GAPS 
    #gap_counts=np.empty(len(fasta_sequences[0].seq),dtype=int)
    
    #gap_counts_w=np.empty(len(fasta_sequences[0].seq))
    #totw=names_table.w.sum()
    #print('loop along fasta')
    #for i in range(len(fasta_sequences[0].seq)):
        #gap_counts[i]=sum(MSA[:,i]=='-')
     #   gap_counts_w[i]=names_table.w[np.where(MSA[:,i]=='-')[0]].sum()
#    MSA_nogap=MSA[:,np.arange(len(fasta_sequences[0].seq))[gap_counts/len(MSA)<=gap_max]] #30% or less gaps per position
    #np.save(path+'gap_counts_w',gap_counts_w)

    ix=names_table.loc[names_table.uniprot_id==pdb_seq].index
    
    i_g=ix[np.argmin(abs(names_table.loc[ix].pos_ini-pdb_uni_beg))] #closest to pdb sequence
    #sel_seqs=MSA[ix,:]!='-'
    #i_g=ix[np.argmax(sel_seqs.sum(axis=1))]
    col_nogap=np.where(MSA[i_g,:]!='-')[0]
    np.save(path+'col_nogap',col_nogap)

    
    
    MSA_nogap=MSA[:,col_nogap] 

    #MSA_nogap=MSA[:,np.arange(len(fasta_sequences[0].seq))[gap_counts_w/totw<=gap_max]] #30% or less gaps per position
    #pd.DataFrame(MSA_nogap).to_csv(path+'MSA_nogap.csv',index_label=False)
    
 #   print('making fasta')
    #save nogap-fasta
    def generator_of_sequences_nat():
        for s,string_seq in enumerate(MSA_nogap):
            yield SeqRecord(Seq(''.join(string_seq)), id=names_table.name[s], 
                            description='cluster='+str(names_table.cluster[s])+' w='+str(names_table.w[s]))

    output_handle=path+'MSA_nogap.fasta'
    SeqIO.write(generator_of_sequences_nat(),output_handle, "fasta")
    
    MSA_nogap_shape=MSA_nogap.shape
    
    del MSA_nogap
    gc.collect()
    MSA_nogap=0
    gc.collect()

#    # MSA 1 PER CLUSTER (FOR SIMPLIFIED DCA)
#     ix=[]
#    names_table_=names_table[~names_table.duplicated('uniprot_id')]
#    for c in names_table.cluster.unique():
#        ix.append(random.choice(names_table_[names_table_.cluster==c].index))
#    # make info, MSA and fasta with selected seq (1 per cluster)
#    pd.DataFrame(MSA_nogap[ix,:]).to_csv(path+'MSA_nogap_1_per_cluster.csv',index_label=False)
#    names_table.iloc[ix].reset_index(drop=True).to_csv('info_alignment_1_per_cluster.csv',index_label=False)

        
    #los numeros que necesito para la tabla los mando aca    
    return family_code,interpro_code,family_name,names_table.name[i_g],cdhit_sim,DATE,MSA_nogap_shape[0],MSA_nogap_shape[1],sum(names_table.w)


def part2(path, family_code):
    
    col_nogap=np.load(path+'col_nogap.npy')
   # npos=len(gap_counts_w) #numero original de posiciones del fasta con gaps

    with open (path+'info_alignment', 'rb') as fp:
         names_table = pickle.load(fp)
   # totw=names_table.w.sum()
   # col_nogap=gap_counts_w/totw<=gap_max

    

    
    # MAKE MSA AND UNIPROT ID (name) LIST FROM FASTA
    
    fastaname=family_code+'_uniprot.txt'   

    fasta_sequences = list(SeqIO.parse(path+fastaname,'fasta'))
    MSA=np.empty([len(fasta_sequences),len(fasta_sequences[0].seq)],dtype='<U1')
    names=[]
    for j in range(len(fasta_sequences)):
        MSA[j,:]=[i for i in fasta_sequences[j].seq]
    MSA=MSA[names_table.msa_index,:] 
    

    
    
    # EXON BOUNDARY POSITIONS TRASLATION TO AA SEQ AND COMPARATION WITH DOMAIN POSITION
    code_table=pd.read_csv(path+'code_table.csv') #400k
    if len(code_table)>2:

        output_file=path+"exon_len_info_all.txt"
        exon_b_full=load_exons(output_file,code_table)
        exon_b_full=exon_b_full.merge(names_table)
        exon_b_full=exon_b_full[~exon_b_full.duplicated(subset='name')].reset_index() # just in case

        exon_b_full['rna-3prot']=exon_b_full.total_bs_len-((exon_b_full.seq_len+1)*3)#sumo uno por el codon stop
        len_error_table=exon_b_full.groupby(['rna-3prot','reading_frame','partial_5','partial_3',
                                             'len_dif','phases_ok']).name.count()
        len_error_table.to_csv(path+'exon_len_error_table.csv')


  #      print('mapping exon boundaries')

        all_bs_aa, ins_bs, intron_ph, last_ex_ph, next_ex_ph, both_ex_ph = translate_exon_pos(exon_b_full,MSA,col_nogap,names_table)
        exon_b_full['exon_bs_final']=all_bs_aa
        exon_b_full['ins_bs']=ins_bs
        exon_b_full['intron_ph_ali']=intron_ph
        exon_b_full['last_ex_ph_ali']=last_ex_ph
        exon_b_full['next_ex_ph_ali']=next_ex_ph
        exon_b_full['both_ex_ph_ali']=both_ex_ph
        
        print(exon_b_full.columns)
        
  #      print('gene weights')

        # reweighting (for genes)
        cluster_size=exon_b_full[~exon_b_full.uniprot_id.duplicated()].groupby(['cluster']).size()
        # not counting double domains in a single sequence to calculate cluster size
        rew=[]
        for i,exons in enumerate(exon_b_full.exon_bs_final):
            rew_i=[]
            for exon in exons:
                rew_i.append(1.0/cluster_size[exon_b_full.cluster[i]])
            rew.append(rew_i)
        exon_b_full['exon_rew']=rew    

  #      print('save exon boundaries data')

        with open(path+'exon_all_file', 'wb') as fp:
            pickle.dump(exon_b_full,fp)
        Output=len(exon_b_full)
    
    else:
        Output=0
    return Output
