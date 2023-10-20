import numpy as np
import pandas as pd
import pickle
import scipy.signal as scs
import urllib
import os
import dca_frustratometer
import random
from Bio import SeqIO
from itertools import combinations

_AA = '-ACDEFGHIKLMNPQRSTVWY'

#B=N
AAdict={'Z':0,'X':0,'-':0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12, 'B':12
        ,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}

def make_domains_from_exons(exon_start,exon_end):
    domains=list(combinations(exon_start+[exon_end[-1]+1],2))
    domain_start=np.transpose(domains)[0]
    domain_end=[x-1 for x in np.transpose(domains)[1]]
    domain_rel_pos=np.transpose([domain_start,domain_end])
    return domain_start,domain_end,domain_rel_pos

def compute_fragment_mask(mask: np.array,
                  fragment_pos: np.array)-> np.array:
    '''
    masks i,j such that:
        i belongs to the fragment, all j
        j belongs to the fragment, all i
    '''
    custom_mask=np.zeros((mask.shape[0],mask.shape[0]),dtype=bool)
    custom_mask[fragment_pos]=True
    custom_mask[:,fragment_pos]=True
    return custom_mask*mask


def compute_fragment_mask_SE(mask: np.array,
                  fragment_pos: np.array)-> np.array:
    '''
    masks i,j such that:
        i,j belongs to the fragment 
    '''
    custom_mask1=np.zeros((mask.shape[0],mask.shape[0]),dtype=bool)
    custom_mask2=np.zeros((mask.shape[0],mask.shape[0]),dtype=bool)

    custom_mask1[:,fragment_pos]=True
    custom_mask2[fragment_pos]=True

    return custom_mask1*custom_mask2*mask
'''
def compute_sequences_energy(seqs: list,
                             potts_model: dict,
                             mask: np.array,
                             split_couplings_and_fields = False 
                             ) -> np.array:
    seq_index = np.array([[_AA.find(aa) for aa in seq] for seq in seqs])
    N_seqs, seq_len = seq_index.shape
    pos_index=np.repeat([np.arange(seq_len)], N_seqs,axis=0)


    pos1=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[0] for p in pos_index])
    pos2=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[1] for p in pos_index])
    aa1=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[0] for s in seq_index])
    aa2=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[1] for s in seq_index])
    
    h = -potts_model['h'][pos_index,seq_index]
    j = -potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * mask

    if split_couplings_and_fields:
        return np.array([h.sum(axis=-1),j_prime.sum(axis=-1).sum(axis=-1) / 2])
    else:
        energy = h.sum(axis=-1) + j_prime.sum(axis=-1).sum(axis=-1) / 2
        return energy
'''
def compute_sequences_energy(seqs: list,
                             potts_model: dict,
                             mask: np.array,
                             split_couplings_and_fields = False ,
                             config_decoys = False,
                             WP = True,
                             fragment_pos = None) -> np.array:
    seq_index = np.array([[_AA.find(aa) for aa in seq] for seq in seqs])
    N_seqs, seq_len = seq_index.shape
    pos_index=np.repeat([np.arange(seq_len)], N_seqs,axis=0)
    
    
    if config_decoys:
        '''
        shuffle index positions for configurational decoys energy calculation
        seqs must be a list of shuffled versions of the native one
        mask MUST BE the original model.mask, not the msa adapted version
        '''
        pos_index=np.array([np.random.choice(pos_index[0],
                                             size=len(pos_index[0]),
                                             replace=False) for x in range(pos_index.shape[0])])
        if WP:
            mask=np.ones(mask.shape)*mask.mean()
            #else: SE, mask must be loaded averaged directly
#           
    if fragment_pos is None:
        h_mask=np.ones(seq_len,dtype=int)
    else:
        h_mask=np.zeros(seq_len,dtype=int)
        h_mask[fragment_pos]=1
        
    
    pos1=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[0] for p in pos_index])
    pos2=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[1] for p in pos_index])
    aa1=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[0] for s in seq_index])
    aa2=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[1] for s in seq_index])
    
    h = -potts_model['h'][pos_index,seq_index]
    j = -potts_model['J'][pos1, pos2, aa1, aa2]
    
    h_prime= h * h_mask
    j_prime = j * mask

    if split_couplings_and_fields:
        return np.array([h_prime.sum(axis=-1),j_prime.sum(axis=-1).sum(axis=-1) / 2])
    else:
        energy = h_prime.sum(axis=-1) + j_prime.sum(axis=-1).sum(axis=-1) / 2
        return energy

def compute_fragment_native_energy_with_interactions(seq: str,
                                                     potts_model: dict,
                                                     mask: np.array,
                                                     fragment_pos: np.array) -> float:
    
    '''
    masks i,j such that:
        i belongs to the fragment, all j
        j belongs to the fragment, all i
    '''
    
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)
    if len(potts_model['J'].shape)==4:
        print('potts')
        h = -potts_model['h'][range(seq_len), seq_index]
        j = -potts_model['J'][pos1, pos2, aa1, aa2]
    else:
        #MJ 
        print('MJ')
        h = 0
        j = -potts_model['J'][aa1, aa2]
    h_mask=np.zeros(seq_len,dtype=int)
    h_mask[fragment_pos]=1
    j_mask=compute_fragment_mask(mask,fragment_pos)                               
    
    h_prime= h*h_mask
    j_prime = j * j_mask

    energy = h_prime.sum() + j_prime.sum() / 2
    return energy

def compute_sequences_fragment_energy_with_interactions(seqs: list,
                                                        potts_model: dict,
                                                        mask: np.array,
                                                        fragment_pos: np.array,
                                                        split_couplings_and_fields = False,
                                                        config_decoys = False,
                                                        msa_mask = 1
                                                        ) -> np.array:

    
    seq_index = np.array([[_AA.find(aa) for aa in seq] for seq in seqs])
    N_seqs, seq_len = seq_index.shape
    pos_index=np.repeat([np.arange(seq_len)], N_seqs,axis=0)
    
    if config_decoys:
        '''
        shuffle index positions for configurational decoys energy calculation
        seqs must be a list of shuffled versions of the native one
        mask MUST BE the original model.mask, not the msa adapted version
        '''
        pos_index=np.array([np.random.choice(pos_index[0],
                                             size=len(pos_index[0]),
                                             replace=False) for x in range(pos_index.shape[0])])
        mask=np.ones(mask.shape)*mask.mean()

    mask=mask*msa_mask    
       
    pos1=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[0] for p in pos_index])
    pos2=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[1] for p in pos_index])
    aa1=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[0] for s in seq_index])
    aa2=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[1] for s in seq_index])
    if len(potts_model['J'].shape)==4:
        h = -potts_model['h'][pos_index,seq_index]
        j = -potts_model['J'][pos1, pos2, aa1, aa2]
    else:
        #MJ 
        h = 0
        j = -potts_model['J'][aa1, aa2]
    h_mask=np.zeros(seq_len,dtype=int)
    h_mask[fragment_pos]=1
    j_mask=compute_fragment_mask(mask,fragment_pos)                               
    
    h_prime= h*h_mask
    j_prime = j * j_mask

    if split_couplings_and_fields:
        return np.array([h_prime.sum(axis=-1),j_prime.sum(axis=-1).sum(axis=-1) / 2])
    else:
        energy = h_prime.sum(axis=-1) + j_prime.sum(axis=-1).sum(axis=-1) / 2
        return energy

#control frustration
def common_exon_len_dist(results_table_pdb,path_,order=10,border=10,threshold=0.01):
    exon_len_=[]
    for family in results_table_pdb.name:

        path_f=path_+family+'/'
        with open (path_f+'pdb_ali_map', 'rb') as fp:
            ali_seq_num_pdb, pdb = pickle.load(fp)    

        pdb_beg=results_table_pdb.real_pdb_beg[results_table_pdb.name==family].values[0]
        npos=len(ali_seq_num_pdb)
        exon_table=pd.read_csv(path_f+'exon_table.csv')
        final_bs,exons_rel_pos,exons,exon_start,exon_end,exons_center,exon_len=get_common_exons(exon_table,
                                                            ali_seq_num_pdb,order,threshold,border,pdb_beg,relative=False)
        exon_len_.append(exon_len)
    return exon_len_

def make_control_fragments(beta,seq_len,min_len):
    # N: ensemble size
    n_exons=int(seq_len*beta*20) # n_exons: much more than expected for a single sequence
    simulated_lengths = np.random.geometric(p=beta, size=n_exons)
    simulated_lengths = simulated_lengths[simulated_lengths>min_len]
    bs=np.cumsum(simulated_lengths)
    bs=bs[bs>min_len]
    bs=bs[bs<(seq_len-min_len)]
    bs=np.hstack([np.zeros(1,int),bs,np.array(seq_len-1)])
    return bs

def get_control_fragments(sim_bs,ali_seq_num_pdb,pdb_beg):
   
    #exon_start=[ali_seq_num_pdb[final_bs[i]] for i in range(len(final_bs[:-1]))]
    #exon_end=[ali_seq_num_pdb[final_bs[i]-1] for i in range(1,len(final_bs))]

    #exons=np.array([exon_start,exon_end]).T


    exon_rel_start=[ali_seq_num_pdb[sim_bs[i]]-pdb_beg+1 for i in range(len(sim_bs[:-1]))]
    exon_rel_end=[ali_seq_num_pdb[sim_bs[i]-1]-pdb_beg+1 for i in range(1,len(sim_bs))]
    sim_exons_rel_pos=np.array([exon_rel_start,exon_rel_end]).T

    sim_exon_len=[(b-a+1) for a,b in sim_exons_rel_pos]

    return sim_exons_rel_pos,sim_exon_len,exon_rel_start,exon_rel_end


def make_decoy_seqs(seq,ndecoys=500):
    decoy_seqs = []
    for i in range(ndecoys):
        decoy_seqs+=[''.join(random.sample(seq, len(seq)))]
    return decoy_seqs 

def find_common_bs(exon_freq,order,thresh,border,npos):
    final_bs=scs.argrelmax(exon_freq,order=order)[0]
    final_bs=final_bs[exon_freq[final_bs]>thresh]
    final_bs=final_bs[final_bs>=border]
    final_bs=final_bs[final_bs<=(npos-border)]
    final_bs=np.hstack([0,final_bs,npos])
    return final_bs

def get_common_exons(exon_table,ali_seq_num_pdb,order,threshold,border,pdb_beg,relative=False):
    npos=len(ali_seq_num_pdb)

    exon_freq=np.histogram(exon_table.exon_bs_aa,bins=np.arange(-0.5,npos+0.5),density=True,
                           weights=exon_table.exon_rew)[0]
    
    final_bs=find_common_bs(exon_freq,order,threshold,border,npos)
    
    #exon_start=[ali_seq_num_pdb[final_bs[i]]-pdb_beg+1 for i in range(len(final_bs[:-1]))]
    #exon_end=[ali_seq_num_pdb[final_bs[i]-1]-pdb_beg+1 for i in range(1,len(final_bs))]
    exon_start=[ali_seq_num_pdb[final_bs[i]]-pdb_beg+1 for i in range(len(final_bs[:-1]))]
    exon_end=[ali_seq_num_pdb[final_bs[i]-1]-pdb_beg+1 for i in range(1,len(final_bs))]
    
    
    exons=np.array([exon_start,exon_end]).T
    exons_center=[(b-a+1)/2+a for a,b in exons]
    exon_len=[(b-a+1) for a,b in exons]
    
    if relative:
        exon_rel_start=[final_bs[i]+pdb_beg-1 for i in range(len(final_bs[:-1]))]
        exon_rel_end=[final_bs[i]-1+pdb_beg-1 for i in range(1,len(final_bs))]
        exons_rel_pos=np.array([exon_rel_start,exon_rel_end]).T
        return final_bs,exons_rel_pos,exons,exon_start,exon_end,exons_center,exon_len
    else:
        exon_rel_start=exon_start
        exon_rel_end=exon_end
        exons_rel_pos=exons
        return final_bs,exons_rel_pos,exons,exon_start,exon_end,exons_center,exon_len
    
    
def compute_model_and_decoys(path,pdb_file,chain,distance_cutoff,sequence_cutoff,ndecoys):
    
    structure=dca_frustratometer.Structure.full_pdb(pdb_file,chain,repair_pdb=False)
    model=dca_frustratometer.AWSEMFrustratometer(structure,
                                             distance_cutoff=distance_cutoff,
                                             sequence_cutoff=sequence_cutoff)
    
    with open(path+'model.pkl', 'wb') as f:
            pickle.dump((model.sequence,model.potts_model,model.mask), f)
    
    decoy_seqs=make_decoy_seqs([x for x in model.sequence],ndecoys=ndecoys)

    with open(path+'decoy_seqs.pkl', 'wb') as f:
            pickle.dump(decoy_seqs, f)

def compute_e_se(ini,fin,pdb_beg,pdb_file,chain,distance_cutoff,sequence_cutoff,decoy_seqs):
    '''
    using resnum 
    '''
    frag_decoy_seqs=[s[(ini-1):(fin)] for s in decoy_seqs]
    structure_exon=dca_frustratometer.Structure.spliced_pdb(pdb_file, chain, 
                                                                    seq_selection='resnum '+str(ini)+'to'+str(fin),
                                                                    repair_pdb=False)
    model_exon=dca_frustratometer.AWSEMFrustratometer(structure_exon,
                                                 distance_cutoff=distance_cutoff,
                                                 sequence_cutoff=sequence_cutoff)
    
    '''
   CONFIG FRUSTRATION SIZE PROBLEM: "EXON POTTS MODEL" IS TOO SHORT TO BE SCRAMBLED
   GENERATING SCRAMBLED VERSIONS OF THE "EXON POTTS MODEL" SEEMS TOO EXPENSIVE TO BE DONE
    '''
    e_decoy_se = compute_sequences_energy(frag_decoy_seqs,model_exon.potts_model,model_exon.mask) #SE
    
    return model_exon.native_energy(),e_decoy_se.mean(),e_decoy_se.std()

## marca

def compute_energy_stats_exons(sequence,potts_model,mask,decoy_seqs,pdb_beg,
                              exon_start,exon_end,exons_rel_pos,msa_mask,config_decoys,
                              pdb_file,chain,distance_cutoff,sequence_cutoff):
    
    e_se_mean,e_se_std,e_wp_mean,e_wp_std,e_native_se,e_native_wp=[],[],[],[],[],[]
    for exon_i in range(len(exon_start)):
        ini,fin=exons_rel_pos[exon_i]
        if fin>potts_model['h'].shape[0]:
            fin=potts_model['h'].shape[0]
        
        print(exon_i,ini,fin)
        fragment_pos=np.arange(ini-1,fin) #por que

        e_native_wp_=compute_fragment_native_energy_with_interactions(sequence,potts_model,mask,fragment_pos)
        e_decoy_wp_ = compute_sequences_fragment_energy_with_interactions(decoy_seqs,potts_model,mask,fragment_pos,
                                                                          split_couplings_and_fields=False,
                                                                         msa_mask=msa_mask,config_decoys=config_decoys) #WP

        
        
        e_native_se_,e_decoy_mean_,e_decoy_std_=compute_e_se(ini,fin,pdb_beg,
                                                          pdb_file,chain,distance_cutoff,sequence_cutoff,
                                                          decoy_seqs)

        e_native_se.append(e_native_se_)
        e_native_wp.append(e_native_wp_)
        e_se_mean.append(e_decoy_mean_)
        e_se_std.append(e_decoy_std_)
        e_wp_mean.append(e_decoy_wp_.mean())
        e_wp_std.append(e_decoy_wp_.std())

    return {'exon':np.arange(len(exon_start)),'exon_start':exon_start,'exon_end':exon_end,'e_native_se':e_native_se,'e_native_wp':e_native_wp,
                'e_se_mean':e_se_mean,'e_se_std':e_se_std,'e_wp_mean':e_wp_mean, 'e_wp_std':e_wp_std}

### MARCA

def compute_energy_stats_exons_macro(results_table_pdb,path_,family,folder,distance_cutoff=8, sequence_cutoff=3,
                                     ctrl_exons=False,N=10, order=10, exon_len_=None,config_decoys=False,domains=False):

    path_f=path_+family+'/'
    pdb_beg=results_table_pdb.real_pdb_beg[results_table_pdb.name==family].values[0]
    chain=results_table_pdb.CHAIN[results_table_pdb.name==family].values[0]
    pdb_code=results_table_pdb.pdb[results_table_pdb.name==family].values[0]
    pdb_file=path_f+pdb_code+'_cleaned.pdb'


    with open(path_f+'model.pkl', 'rb') as f:
                sequence,potts_model,mask = pickle.load(f)
    with open(path_f+'decoy_seqs.pkl', 'rb') as f:
                decoy_seqs = pickle.load(f)
    with open (path_f+'pdb_ali_map', 'rb') as fp:
            ali_seq_num_pdb, pdb = pickle.load(fp)    
        
    if family=='SNase':
            print('pdb incomplete')
            ali_seq_num_pdb=ali_seq_num_pdb[:-1]
            
    npos=len(ali_seq_num_pdb)

    msa_mask,len_pre,len_post,cut_pre,cut_post=create_msa_mask(mask,ali_seq_num_pdb,pdb_beg)
    
    if ctrl_exons:
        seq_len=npos
        min_len=order
        exon_len_all=np.hstack([x[:-1] for x in exon_len_])
        beta=1/exon_len_all.mean()
        for n in range(N):
    
    # simulating 1 set of exons:

            exons_rel_pos,exon_len,exon_start,exon_end=get_control_fragments(make_control_fragments(beta,
                                                                                                   seq_len,
                                                                                                   min_len),
                                                                            ali_seq_num_pdb+1,pdb_beg)
        
            ###########
            if domains:
                domain_start,domain_end,domain_rel_pos=make_domains_from_exons(exon_start,exon_end)
                result=compute_energy_stats_exons(sequence,potts_model,mask,decoy_seqs,pdb_beg,
                                      domain_start,domain_end,domain_rel_pos,
                                      msa_mask,config_decoys,
                                                 pdb_file,chain,distance_cutoff,sequence_cutoff)


                
                
    #        os.system('mkdir '+path_f+folder)
                with open(path_f+folder+'/domains_ctrl_energies'+str(n)+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                result=compute_energy_stats_exons(sequence,potts_model,mask,decoy_seqs,pdb_beg,
                                  exon_start,exon_end,exons_rel_pos,
                                  msa_mask,config_decoys,
                                                 pdb_file,chain,distance_cutoff,sequence_cutoff)
    
#        os.system('mkdir '+path_f+folder)
                with open(path_f+folder+'/ctrl_energies'+str(n)+'.pkl', 'wb') as f:
                    pickle.dump(result, f)

        
    else:
        order=10
        threshold=0.01
        border=order

        exon_table=pd.read_csv(path_f+'exon_table.csv')
        final_bs,exons_rel_pos,exons,exon_start,exon_end,exons_center,exon_len=get_common_exons(exon_table,
                                                            ali_seq_num_pdb+1,order,threshold,border,pdb_beg,relative=False)
        

        if domains:
            domain_start,domain_end,domain_rel_pos=make_domains_from_exons(exon_start,exon_end)
            result=compute_energy_stats_exons(sequence,potts_model,mask,decoy_seqs,pdb_beg,
                                  domain_start,domain_end,domain_rel_pos,msa_mask,config_decoys,
                                             pdb_file,chain,distance_cutoff,sequence_cutoff)
                
            os.system('mkdir '+path_f+folder)
            with open(path_f+folder+'/domains_energies.pkl', 'wb') as f:
                pickle.dump(result, f)

        else:
            result=compute_energy_stats_exons(sequence,potts_model,mask,decoy_seqs,pdb_beg,
                                  exon_start,exon_end,exons_rel_pos,msa_mask,config_decoys,
                                             pdb_file,chain,distance_cutoff,sequence_cutoff)
    
            os.system('mkdir '+path_f+folder)
            with open(path_f+folder+'/exon_energies.pkl', 'wb') as f:
                pickle.dump(result, f)
            
    return 0




def compute_energy_sliding_window(sequence,potts_model,mask,msa_mask,decoy_seqs,win_size,positions,
                              pdb_file,chain,distance_cutoff,sequence_cutoff,config_decoys):
    
    e_wp_mean,e_wp_std,e_native_wp,win_size_array=[],[],[],[]
    dif=(win_size-1)//2
    positions_=positions[dif:-dif]
    for i in positions_:
        ini,fin=i-dif,i+dif
        print(i,ini,fin)
        fragment_pos=np.arange(ini-1,fin) 

        e_native_wp_=compute_fragment_native_energy_with_interactions(sequence,potts_model,mask*msa_mask,fragment_pos)
        e_decoy_wp_ = compute_sequences_fragment_energy_with_interactions(decoy_seqs,potts_model,mask,fragment_pos,
                                                                         config_decoys,msa_mask=msa_mask) #WP

        win_size_array.append(win_size)
        e_native_wp.append(e_native_wp_)
        e_wp_mean.append(e_decoy_wp_.mean())
        e_wp_std.append(e_decoy_wp_.std())

    return {'frag_center':positions_,'win_size':win_size_array,'e_native_wp':e_native_wp,
                'e_wp_mean':e_wp_mean, 'e_wp_std':e_wp_std}


def sliding_window_macro(results_table_pdb,path_,family,folder,distance_cutoff=8, sequence_cutoff=3,
                                     order=10,win_size=5,config_decoys=False):

    path_f=path_+family+'/'
    pdb_beg=results_table_pdb.real_pdb_beg[results_table_pdb.name==family].values[0]
    chain=results_table_pdb.CHAIN[results_table_pdb.name==family].values[0]
    pdb_code=results_table_pdb.pdb[results_table_pdb.name==family].values[0]
    pdb_file=path_f+pdb_code+'_cleaned.pdb'


    with open(path_f+'model.pkl', 'rb') as f:
                sequence,potts_model,mask = pickle.load(f)
    with open(path_f+'decoy_seqs.pkl', 'rb') as f:
        decoy_seqs = pickle.load(f)
    
    with open (path_f+'pdb_ali_map', 'rb') as fp:
            ali_seq_num_pdb, pdb = pickle.load(fp)    
            
    positions=ali_seq_num_pdb-pdb_beg+1
    
            #using mask_ as in MSA frustra!!!!
    msa_mask,len_pre,len_post,cut_pre,cut_post=create_msa_mask(mask,ali_seq_num_pdb,pdb_beg)
    
    if config_decoys:
        
        decoy_type='_av_configurational'
        
    else:
        decoy_type=''#mutational

        

    
    result=compute_energy_sliding_window(sequence,potts_model,mask,msa_mask,decoy_seqs,win_size,positions,
                              pdb_file,chain,distance_cutoff,sequence_cutoff,config_decoys)

        
    os.system('mkdir '+path_f+folder)
    with open(path_f+folder+'/window_'+str(win_size)+'_energies_mask'+decoy_type+'.pkl', 'wb') as f:
        pickle.dump(result, f)
            
    return 0

#################### MSA func

def make_weighted_subset(MSA,names,cluster,Nsubset=1000):
    df=pd.DataFrame({'uniprot_name':names,'cluster':cluster,'id':range(len(cluster))})
    idx=df.groupby(['cluster'])['id'].apply(np.random.choice).values
    if Nsubset<len(idx):
        idx_subset=np.random.choice(idx,Nsubset,False)
    else:
        idx_subset=idx
    MSA_subset=MSA[idx_subset,:]
    names_subset=df.loc[idx_subset]
    return MSA_subset,names_subset

def make_exon_subset(MSA,names,cluster,exon_table,Nsubset=1000):
    df=pd.DataFrame({'uniprot_name':names,'cluster':cluster,'id':range(len(cluster))})
    exon_table_MSA=exon_table.merge(df)

    Nclusters=len(exon_table_MSA.cluster.unique())
    if Nclusters<=Nsubset:
        Nsubset=Nclusters
    
    idx=exon_table_MSA.groupby(['cluster'])['id'].apply(np.random.choice).values
    idx_subset=np.random.choice(idx,Nsubset, False)

    MSA_exon_bs=MSA[idx_subset,:]
    names_subset=df.loc[idx_subset]
    exon_table_MSA=exon_table.merge(names_subset)   
    
    return MSA_exon_bs,exon_table_MSA

def concatenate_MSA(MSA_,len_pre,len_post):
    if len_pre>0:
        arr_pre=np.array([['-']*MSA_.shape[0]]*len_pre)
        aux=np.concatenate([arr_pre,MSA_.T])
    else:
        aux=MSA_.T
    if len_post>0:
        arr_post=np.array([['-']*MSA_.shape[0]]*len_post)
        MSA_concat=np.concatenate([aux,arr_post]).T
    else:
        MSA_concat=aux.T
    return np.char.upper(MSA_concat)

def create_msa_mask_OLD(mask,ali_seq_num_pdb,pdb_beg):
    mask_pos=ali_seq_num_pdb-pdb_beg
    msa_mask_=np.ones(mask.shape,dtype=bool)
    #print(mask_pos,mask.shape)
    if mask_pos[0]<0:
        cut_pre=-mask_pos[0]
        len_pre=0
    else:
        msa_mask_[0:(mask_pos)[0]][:]=False
        len_pre=len(msa_mask_[0:(mask_pos)[0]][:])
        cut_pre=0
    if (mask_pos)[-1]>msa_mask_.shape[0]:
        cut_post=msa_mask_.shape[0]-(mask_pos)[-1]
        len_post=0
    else:
        msa_mask_[:][((mask_pos)[-1]+1):]=False
        len_post=len(msa_mask_[:][((mask_pos)[-1]+1):])
        cut_post=0

    msa_mask=msa_mask_*msa_mask_.T
    print(len_pre,len_post,cut_pre,cut_post)
    return mask*msa_mask,len_pre,len_post,cut_pre,cut_post

# prepared for config decoys computation: RETURN ONLY MSA_MASK, NOT THE PRODUCT
def create_msa_mask(mask,ali_seq_num_pdb,pdb_beg):
    mask_pos=ali_seq_num_pdb-pdb_beg
    msa_mask_=np.ones(mask.shape,dtype=bool)
    #print(mask_pos,mask.shape)
    if mask_pos[0]<0:
        cut_pre=-mask_pos[0]
        len_pre=0
    else:
        msa_mask_[0:(mask_pos)[0]][:]=False
        len_pre=len(msa_mask_[0:(mask_pos)[0]][:])
        cut_pre=0
    if (mask_pos)[-1]>msa_mask_.shape[0]:
        cut_post=msa_mask_.shape[0]-(mask_pos)[-1]
        len_post=0
    else:
        msa_mask_[:][((mask_pos)[-1]+1):]=False
        len_post=len(msa_mask_[:][((mask_pos)[-1]+1):])
        cut_post=0

    msa_mask=msa_mask_*msa_mask_.T
    print(len_pre,len_post,cut_pre,cut_post)
    return msa_mask,len_pre,len_post,cut_pre,cut_post


def cut_msa(MSA_,cut_pre,cut_post):
    if cut_post>0:
        return MSA_[:,cut_pre:-cut_post]
    else:
        return MSA_[:,cut_pre:]

    
def cut_positions(positions,last_pos=None):
    positions+=-1
    cut_pre,cut_pos=0,0
    if positions[0]<0:
        cut_pre=positions[0]
        positions=positions[-cut_pre:]
    if last_pos is not None:
        if positions[-1]>last_pos:
            cut_pos=(last_pos-positions[-1])
            positions=positions[:-cut_pos]
    return positions+1,cut_pre,cut_pos

def compute_sequences_fragment_energy_with_interactions_ix(seq_index: np.array,
                                                        potts_model: dict,
                                                        mask: np.array,
                                                        fragment_pos: np.array,
                                                        split_couplings_and_fields = False
                                                        ) -> np.array:
    #seq_index = np.array([[_AA.find(aa) for aa in seq] for seq in seqs])
    N_seqs, seq_len = seq_index.shape
    pos_index=np.repeat([np.arange(seq_len)], N_seqs,axis=0)

    
    pos1=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[0] for p in pos_index])
    pos2=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[1] for p in pos_index])
    aa1=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[0] for s in seq_index])
    aa2=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[1] for s in seq_index])
    
    h = -potts_model['h'][pos_index,seq_index]
    j = -potts_model['J'][pos1, pos2, aa1, aa2]
    
    h_mask=np.zeros(seq_len,dtype=int)
    h_mask[fragment_pos]=1
    j_mask=compute_fragment_mask(mask,fragment_pos)                               
    
    h_prime= h*h_mask
    j_prime = j * j_mask

    if split_couplings_and_fields:
        return np.array([h_prime.sum(axis=-1),j_prime.sum(axis=-1).sum(axis=-1) / 2])
    else:
        energy = h_prime.sum(axis=-1) + j_prime.sum(axis=-1).sum(axis=-1) / 2
        return energy


def compute_energy_sliding_window_MSA(seq_index,potts_model,mask,win_size,positions):                             
    e_wp_mean,e_wp_std,e_native_wp,win_size_array=[],[],[],[]
    dif=(win_size-1)//2
    positions_=positions[dif:-dif]
    result_table=np.zeros((len(positions_),seq_index.shape[0]+2))
    result_table[:,0]=positions_
    #print(positions_)
    for i_,i in enumerate(positions_):
        ini,fin=i-dif,i+dif
        #print(i,ini,fin)
        fragment_pos=np.arange(ini-1,fin) 

        result_table[i_,2:] = compute_sequences_fragment_energy_with_interactions_ix(seq_index,
                                                                                  potts_model,mask,
                                                                                  fragment_pos) #WP
        result_table[i_,1]=win_size

    
    return result_table
                


def sliding_window_macro_MSA(results_table_pdb,path_,family,folder,win_size=5,make_exon_subset=False):

    path_f=path_+family+'/'
    pdb_beg=results_table_pdb.real_pdb_beg[results_table_pdb.name==family].values[0]
    family_code=results_table_pdb.pfam[results_table_pdb.name==family].values[0]


    with open(path_f+'model.pkl', 'rb') as f:
                sequence,potts_model,mask = pickle.load(f)
    
    with open (path_f+'pdb_ali_map', 'rb') as fp:
            ali_seq_num_pdb, pdb = pickle.load(fp)    
            
    positions=ali_seq_num_pdb-pdb_beg+1
    
    ################################ 

    col_nogap=np.load(path_f+'col_nogap.npy')

    #with open (path_f+'info_alignment', 'rb') as fp:
    #     names_table = pickle.load(fp)
    names_table=pd.read_pickle(path_f+'info_alignment')
    # MAKE MSA AND UNIPROT ID (name) LIST FROM FASTA
    fastaname=family_code+'_uniprot.txt'   

    fasta_sequences = list(SeqIO.parse(path_f+fastaname,'fasta'))
    MSA=np.empty([len(fasta_sequences),len(fasta_sequences[0].seq)],dtype='<U1')
    names=[]

    for j in range(len(fasta_sequences)):
        MSA[j,:]=[i for i in fasta_sequences[j].seq]
    names=names_table.name.values

    MSA=MSA[names_table.msa_index,:]     
    MSA=MSA[:,col_nogap]
    
    w=names_table.w.values
    cluster=names_table.cluster.values
    ################################
    
    msa_mask,len_pre,len_post,cut_pre,cut_post=create_msa_mask(mask,ali_seq_num_pdb,pdb_beg)
    mask_=mask*msa_mask
    
    # exon subset
    if make_exon_subset:
        exon_table=pd.read_csv(path_f+'exon_table.csv')    
        MSA_exon_bs_,exon_table_MSA=make_exon_subset(MSA,names,cluster,exon_table)
        MSA_exon_bs=cut_msa(MSA_exon_bs_,cut_pre,cut_post)
        print(family,MSA_exon_bs.shape)
        MSA_concat=concatenate_MSA(MSA_exon_bs,len_pre,len_post)
        if mask.shape[1]!=MSA_concat.shape[1]:
            print(mask.shape[1],MSA_concat.shape[1])
            print('wrong shape')
        seq_index=pd.DataFrame(MSA_concat).apply(lambda x: x.map(AAdict)).values
        result=compute_energy_sliding_window_MSA(seq_index,potts_model,mask_,win_size,positions)
        os.system('mkdir '+path_f+folder)
        with open(path_f+folder+'/window_'+str(win_size)+'_energies_msa_exon.pkl', 'wb') as f:
            pickle.dump(result, f)
        with open(path_f+folder+'/table_exon.pkl', 'wb') as f:
            pickle.dump(exon_table_MSA, f)
            
    # general subset     
    MSA_subset_,names_subset=make_weighted_subset(MSA,names,cluster,Nsubset=1000)
    MSA_subset=cut_msa(MSA_subset_,cut_pre,cut_post)
    print(family,MSA_subset.shape)
    MSA_concat2=concatenate_MSA(MSA_subset,len_pre,len_post)
    if mask.shape[1]!=MSA_concat2.shape[1]:
        print(mask.shape[1],MSA_concat2.shape[1])
        print('wrong shape')
    seq_index2=pd.DataFrame(MSA_concat2).apply(lambda x: x.map(AAdict)).values
    result2=compute_energy_sliding_window_MSA(seq_index2,potts_model,mask_,win_size,positions)
    os.system('mkdir '+path_f+folder)
    with open(path_f+folder+'/window_'+str(win_size)+'_energies_msa_subset.pkl', 'wb') as f:
        pickle.dump(result2, f)
    
    return 0
######################## FOLDABILITY 

def compute_energy_cleavage_point(sequence,potts_model,mask,msa_mask,decoy_seqs,min_d,positions,config_decoys):
    
    e_wp_meana,e_wp_stda,e_native_wpa,len_a=[],[],[],[]
    e_wp_meanb,e_wp_stdb,e_native_wpb,len_b=[],[],[],[]
    
    # to avoid ini<0,fin>length
    positions,cut_pre,cut_pos=cut_positions(positions,last_pos=mask.shape[0])

    positions_=positions[min_d:-min_d]
    for i in positions_:
        ini_a,fin_a=positions[0],i
        ini_b,fin_b=i+1,positions[-1]
        print(i,ini_a,fin_a,ini_b,fin_b)
        fragment_pos_a=np.arange(ini_a-1,fin_a) 
        fragment_pos_b=np.arange(ini_b-1,fin_b)
        
        e_native_wp_a=compute_fragment_native_energy_with_interactions(sequence,potts_model,mask*msa_mask,fragment_pos_a)
        e_decoy_wp_a = compute_sequences_fragment_energy_with_interactions(decoy_seqs,potts_model,mask,fragment_pos_a,
                                                                          config_decoys=config_decoys,msa_mask=msa_mask) #WP

        e_native_wp_b=compute_fragment_native_energy_with_interactions(sequence,potts_model,mask*msa_mask,fragment_pos_b)
        e_decoy_wp_b = compute_sequences_fragment_energy_with_interactions(decoy_seqs,potts_model,mask,fragment_pos_b,
                                                                          config_decoys=config_decoys,msa_mask=msa_mask) #WP
        
        e_native_wpa.append(e_native_wp_a)
        e_wp_meana.append(e_decoy_wp_a.mean())
        e_wp_stda.append(e_decoy_wp_a.std())

        e_native_wpb.append(e_native_wp_b)
        e_wp_meanb.append(e_decoy_wp_b.mean())
        e_wp_stdb.append(e_decoy_wp_b.std())

        len_a.append(fin_a-ini_a+1)
        len_b.append(fin_b-ini_b+1)
    f_a=(np.array(e_native_wpa)-np.array(e_wp_meana))/np.array(e_wp_stda)
    f_b=(np.array(e_native_wpb)-np.array(e_wp_meanb))/np.array(e_wp_stdb)
    fold_a=f_a/np.sqrt(np.array(len_a))
    fold_b=f_b/np.sqrt(np.array(len_b))
    fold_av=(fold_a+fold_b)/2
    return {'cleavage_point':positions_,'N_a':len_a,'N_b':len_b,'e_native_wp_a':e_native_wpa,
                'e_wp_mean_a':e_wp_meana, 'e_wp_std_a':e_wp_stda,'f_a':f_a,
                'fold_a':fold_a,
                'e_native_wp_b':e_native_wpb,
                'e_wp_mean_b':e_wp_meanb, 'e_wp_std_a':e_wp_stdb,
                'f_b':f_b,
                'fold_b':fold_b,
                'fold_av':fold_av}
'''
def cleavage_point_macro(results_table_pdb,path_,family,folder,distance_cutoff=8, sequence_cutoff=3,
                                     min_d=5,config_decoys=False):

    path_f=path_+family+'/'
    pdb_beg=results_table_pdb.real_pdb_beg[results_table_pdb.name==family].values[0]
    chain=results_table_pdb.CHAIN[results_table_pdb.name==family].values[0]
    pdb_code=results_table_pdb.pdb[results_table_pdb.name==family].values[0]
    pdb_file=path_f+pdb_code+'_cleaned.pdb'


    with open(path_f+'model.pkl', 'rb') as f:
                sequence,potts_model,mask = pickle.load(f)
    with open(path_f+'decoy_seqs.pkl', 'rb') as f:
        decoy_seqs = pickle.load(f)
    
    with open (path_f+'pdb_ali_map', 'rb') as fp:
            ali_seq_num_pdb, pdb = pickle.load(fp)    
            
    positions=ali_seq_num_pdb-pdb_beg+1
    
    #using mask_ as in MSA frustra!!!!
    msa_mask,len_pre,len_post,cut_pre,cut_post=create_msa_mask(mask,ali_seq_num_pdb,pdb_beg)
    
    result=compute_energy_cleavage_point(sequence,potts_model,mask,msa_mask,decoy_seqs,min_d,positions,config_decoys)
       

    os.system('mkdir '+path_f+folder)
    with open(path_f+folder+'/foldability_config.pkl', 'wb') as f:
        pickle.dump(result, f)
            
    return 0
'''
##############  frustra total SE/WP with REAL EXONS

def compute_sequences_energy_ix(seq_index: np.array,
                             potts_model: dict,
                             mask: np.array,
                             split_couplings_and_fields = False ,
                             config_decoys = False
                             ) -> np.array:
    #seq_index = np.array([[_AA.find(aa) for aa in seq] for seq in seqs])
    N_seqs, seq_len = seq_index.shape
    pos_index=np.repeat([np.arange(seq_len)], N_seqs,axis=0)
    
    
    if config_decoys:
        '''
        shuffle index positions for configurational decoys energy calculation
        seqs must be a list of shuffled versions of the native one
        mask MUST BE the original model.mask, not the msa adapted version
        '''
        pos_index=np.array([np.random.choice(pos_index[0],
                                             size=len(pos_index[0]),
                                             replace=False) for x in range(pos_index.shape[0])])
        mask=np.ones(mask.shape)*mask.mean()


    pos1=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[0] for p in pos_index])
    pos2=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[1] for p in pos_index])
    aa1=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[0] for s in seq_index])
    aa2=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[1] for s in seq_index])
    
    h = -potts_model['h'][pos_index,seq_index]
    j = -potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * mask

    if split_couplings_and_fields:
        return np.array([h.sum(axis=-1),j_prime.sum(axis=-1).sum(axis=-1) / 2])
    else:
        energy = h.sum(axis=-1) + j_prime.sum(axis=-1).sum(axis=-1) / 2
        return energy
    
def compute_e_se_MSA(ini,fin,pdb_file,chain,distance_cutoff,sequence_cutoff,decoy_seqs,exon_seq_index):
    '''
    using resnum 
    '''
    
    frag_decoy_seqs=[s[(ini-1):(fin)] for s in decoy_seqs]
    structure_exon=dca_frustratometer.Structure.spliced_pdb(pdb_file, chain, 
                                                            seq_selection='resnum '+str(ini)+'to'+str(fin),
                                                            repair_pdb=False)
    model_exon=dca_frustratometer.AWSEMFrustratometer(structure_exon,
                                                 distance_cutoff=distance_cutoff,
                                                 sequence_cutoff=sequence_cutoff)
    
    '''
   CONFIG FRUSTRATION SIZE PROBLEM: "EXON POTTS MODEL" IS TOO SHORT TO BE SCRAMBLED
   GENERATING SCRAMBLED VERSIONS OF THE "EXON POTTS MODEL" SEEMS TOO EXPENSIVE TO BE DONE
    '''
    e_decoy_se = compute_sequences_energy(frag_decoy_seqs,model_exon.potts_model,model_exon.mask) #SE
    
    e_native_se = compute_sequences_energy_ix(exon_seq_index,model_exon.potts_model,model_exon.mask) #SE
    
    return e_native_se,e_decoy_se.mean(),e_decoy_se.std()

def compute_e_se_MSA(ini,fin,pdb_file,chain,distance_cutoff,sequence_cutoff,decoy_seqs,exon_seq_index):
    '''
    using resnum 
    '''
    
    frag_decoy_seqs=[s[(ini-1):(fin)] for s in decoy_seqs]
    structure_exon=dca_frustratometer.Structure.spliced_pdb(pdb_file, chain, 
                                                            seq_selection='resnum '+str(ini)+'to'+str(fin),
                                                            repair_pdb=False)
    model_exon=dca_frustratometer.AWSEMFrustratometer(structure_exon,
                                                 distance_cutoff=distance_cutoff,
                                                 sequence_cutoff=sequence_cutoff)
    
    '''
   CONFIG FRUSTRATION SIZE PROBLEM: "EXON POTTS MODEL" IS TOO SHORT TO BE SCRAMBLED
   GENERATING SCRAMBLED VERSIONS OF THE "EXON POTTS MODEL" SEEMS TOO EXPENSIVE TO BE DONE
    '''
    e_decoy_se = compute_sequences_energy(frag_decoy_seqs,model_exon.potts_model,model_exon.mask) #SE
    
    e_native_se = compute_sequences_energy_ix(exon_seq_index,model_exon.potts_model,model_exon.mask) #SE
    
    return e_native_se,e_decoy_se.mean(),e_decoy_se.std()


def compute_energy_stats_MSA_exon(seq_index,ini,fin,potts_model,mask,decoy_seqs,
                                  msa_mask,pdb_file,chain,distance_cutoff,
                                  sequence_cutoff,config_decoys=False):
    


    fragment_pos=np.arange(ini-1,fin) 

    e_native_wp_MSA=compute_sequences_fragment_energy_with_interactions_ix(seq_index,potts_model,mask,fragment_pos)
    e_decoy_wp =compute_sequences_fragment_energy_with_interactions(decoy_seqs,potts_model,mask,fragment_pos,
                                                                      split_couplings_and_fields=False,
                                                                     msa_mask=msa_mask,config_decoys=config_decoys) #WP



    e_native_se_MSA,e_decoy_mean_se,e_decoy_std_se=compute_e_se_MSA(ini,fin,
                                                                    pdb_file,
                                                                    chain,distance_cutoff,sequence_cutoff,
                                                                    decoy_seqs,seq_index[:,fragment_pos])
    
    return {'e_native_wp_MSA':e_native_wp_MSA,
            'e_decoy_wp_mean':np.repeat(e_decoy_wp.mean(),len(e_native_wp_MSA)),
            'e_decoy_wp_std':np.repeat(e_decoy_wp.std(),len(e_native_wp_MSA)),
            'e_native_se_MSA':e_native_se_MSA,
            'e_decoy_se_mean':np.repeat(e_decoy_mean_se,len(e_native_wp_MSA)),
            'e_decoy_se_std':np.repeat(e_decoy_std_se,len(e_native_wp_MSA))}


######################## FOLDABILITY SE
def compute_e_se_config(ini,fin,decoy_seqs,mask,potts_model):
    '''
    using resnum 
    '''
    mask=np.ones(mask.shape)*mask.mean()
        
    # strictly mask the mean mask to fit this fragment 
    fragment_pos=np.arange(ini-1,fin) 
    mask_=compute_fragment_mask_SE(mask,fragment_pos)
        
    e_decoy_se = compute_sequences_energy(decoy_seqs,potts_model,mask_,config_decoys = True,
                             WP = False,fragment_pos=fragment_pos) #SE
    
    return e_decoy_se.mean(),e_decoy_se.std()

def compute_energy_cleavage_point_SE(pdb_beg,pdb_file,chain,distance_cutoff,sequence_cutoff,decoy_seqs,
                                     min_d,positions,config_decoys=False,mask=None,potts_model=None):
    
    e_se_meana,e_se_stda,e_native_sea,len_a=[],[],[],[]
    e_se_meanb,e_se_stdb,e_native_seb,len_b=[],[],[],[]
    
    if mask is not None:
        last_pos=mask.shape[0]
    else:
        last_pos=None
    positions,cut_pre,cut_pos=cut_positions(positions,last_pos=last_pos) 

    positions_=positions[min_d:-min_d]
    for i in positions_:
        ini_a,fin_a=positions[0],i
        ini_b,fin_b=i+1,positions[-1]
        print(i,ini_a,fin_a,ini_b,fin_b)
        
        if config_decoys:
            e_decoy_mean_a,e_decoy_std_a=compute_e_se_config(ini_a,fin_a,decoy_seqs,mask,potts_model)
            e_decoy_mean_b,e_decoy_std_b=compute_e_se_config(ini_b,fin_b,decoy_seqs,mask,potts_model)
            e_native_se_a=np.nan
            e_native_se_b=np.nan
            
        else:
            e_native_se_a,e_decoy_mean_a,e_decoy_std_a=compute_e_se(ini_a,fin_a,pdb_beg,pdb_file,chain,
                                                                    distance_cutoff,sequence_cutoff,decoy_seqs)
            e_native_se_b,e_decoy_mean_b,e_decoy_std_b=compute_e_se(ini_b,fin_b,pdb_beg,pdb_file,chain,
                                                                    distance_cutoff,sequence_cutoff,decoy_seqs)

        
        e_native_sea.append(e_native_se_a)
        e_se_meana.append(e_decoy_mean_a)
        e_se_stda.append(e_decoy_std_a)

        e_native_seb.append(e_native_se_b)
        e_se_meanb.append(e_decoy_mean_b)
        e_se_stdb.append(e_decoy_std_b)

        len_a.append(fin_a-ini_a+1)
        len_b.append(fin_b-ini_b+1)
    f_a=(np.array(e_native_sea)-np.array(e_se_meana))/np.array(e_se_stda)
    f_b=(np.array(e_native_seb)-np.array(e_se_meanb))/np.array(e_se_stdb)
    fold_a=f_a/np.sqrt(np.array(len_a))
    fold_b=f_b/np.sqrt(np.array(len_b))
    fold_av=(fold_a+fold_b)/2
    if config_decoys:
        result={'cleavage_point':positions_,'N_a':len_a,'N_b':len_b,
                'e_se_mean_a_config':e_se_meana, 'e_se_std_a_config':e_se_stda,
                'e_se_mean_b_config':e_se_meanb, 'e_se_std_b_config':e_se_stdb}
        
    else:
        result={'cleavage_point':positions_,'N_a':len_a,'N_b':len_b,'e_native_se_a':e_native_sea,
                'e_se_mean_a':e_se_meana, 'e_se_std_a':e_se_stda,'f_a':f_a,
                'fold_a':fold_a,
                'e_native_se_b':e_native_seb,
                'e_se_mean_b':e_se_meanb, 'e_se_std_b':e_se_stdb,
                'f_b':f_b,
                'fold_b':fold_b,
                'fold_av':fold_av}
    return result



def cleavage_point_macro(results_table_pdb,path_,family,folder,distance_cutoff=8, sequence_cutoff=3,
                                     min_d=5,config_decoys=False,WP=True):

    path_f=path_+family+'/'
    pdb_beg=results_table_pdb.real_pdb_beg[results_table_pdb.name==family].values[0]
    chain=results_table_pdb.CHAIN[results_table_pdb.name==family].values[0]
    pdb_code=results_table_pdb.pdb[results_table_pdb.name==family].values[0]
    pdb_file=path_f+pdb_code+'_cleaned.pdb'


    with open(path_f+'decoy_seqs.pkl', 'rb') as f:
        decoy_seqs = pickle.load(f)
    
    with open (path_f+'pdb_ali_map', 'rb') as fp:
        ali_seq_num_pdb, pdb = pickle.load(fp)    
            
    positions=ali_seq_num_pdb-pdb_beg+1
    
    #using mask_ as in MSA frustra!!!!
    if WP:
        with open(path_f+'model.pkl', 'rb') as f:
                sequence,potts_model,mask = pickle.load(f)
        msa_mask,len_pre,len_post,cut_pre,cut_post=create_msa_mask(mask,ali_seq_num_pdb,pdb_beg)

        result=compute_energy_cleavage_point(sequence,potts_model,mask,msa_mask,decoy_seqs,
                                             min_d,positions,config_decoys)
        str_wpse=''
    else: #SE
        str_wpse='_se'

        if config_decoys:
            with open(path_f+'model.pkl', 'rb') as f:
                sequence,potts_model,mask = pickle.load(f)
            result=compute_energy_cleavage_point_SE(pdb_beg,pdb_file,chain,distance_cutoff,sequence_cutoff,
                                                    decoy_seqs,min_d,positions,config_decoys=config_decoys,
                                                    mask=mask,potts_model=potts_model)
            
        else:

            result=compute_energy_cleavage_point_SE(pdb_beg,pdb_file,chain,distance_cutoff,sequence_cutoff,
                                                    decoy_seqs,min_d,positions,config_decoys=config_decoys,
                                                    mask=None,potts_model=None)
            
    if config_decoys:
        str_d='_config'
    else:
        str_d=''
        
    os.system('mkdir '+path_f+folder)
    with open(path_f+folder+'/foldability'+str_d+str_wpse+'.pkl', 'wb') as f:
        pickle.dump(result, f)
            
    return result


############### real exon total frustration ###############
def compute_energy_stats_MSA_exon(seq_index,ini,fin,potts_model,mask,decoy_seqs,
                                  msa_mask,pdb_file,chain,distance_cutoff,
                                  sequence_cutoff,config_decoys=False):
    


    fragment_pos=np.arange(ini-1,fin) 
    #WP
    e_native_wp_MSA=compute_sequences_fragment_energy_with_interactions_ix(seq_index,potts_model,mask,fragment_pos)
    e_decoy_wp =compute_sequences_fragment_energy_with_interactions(decoy_seqs,potts_model,mask,fragment_pos,
                                                                      split_couplings_and_fields=False,
                                                                     msa_mask=msa_mask,config_decoys=config_decoys) #WP


    #SE
    if config_decoys:
        e_native_se_MSA=compute_e_se_MSA_native(ini,fin,pdb_file,chain,distance_cutoff,sequence_cutoff,
                                                seq_index[:,fragment_pos])
        
        e_decoy_mean_se,e_decoy_std_se=compute_e_se_config_MSA(ini,fin,decoy_seqs,mask,
                                                               potts_model)
    else:
        e_native_se_MSA,e_decoy_mean_se,e_decoy_std_se=compute_e_se_MSA(ini,fin,
                                                                    pdb_file,
                                                                    chain,distance_cutoff,sequence_cutoff,
                                                                    decoy_seqs,seq_index[:,fragment_pos])
    
    return {'e_native_wp_MSA':e_native_wp_MSA,
            'e_decoy_wp_mean':np.repeat(e_decoy_wp.mean(),len(e_native_wp_MSA)),
            'e_decoy_wp_std':np.repeat(e_decoy_wp.std(),len(e_native_wp_MSA)),
            'e_native_se_MSA':e_native_se_MSA,
            'e_decoy_se_mean':np.repeat(e_decoy_mean_se,len(e_native_wp_MSA)),
            'e_decoy_se_std':np.repeat(e_decoy_std_se,len(e_native_wp_MSA))}


def compute_e_se_MSA_native(ini,fin,pdb_file,chain,distance_cutoff,sequence_cutoff,exon_seq_index):
    '''
    using resnum 
    '''
    
    structure_exon=dca_frustratometer.Structure.spliced_pdb(pdb_file, chain, 
                                                            seq_selection='resnum '+str(ini)+'to'+str(fin),
                                                            repair_pdb=False)
    model_exon=dca_frustratometer.AWSEMFrustratometer(structure_exon,
                                                 distance_cutoff=distance_cutoff,
                                                 sequence_cutoff=sequence_cutoff)
    
    '''
   CONFIG FRUSTRATION SIZE PROBLEM: "EXON POTTS MODEL" IS TOO SHORT TO BE SCRAMBLED
   GENERATING SCRAMBLED VERSIONS OF THE "EXON POTTS MODEL" SEEMS TOO EXPENSIVE TO BE DONE
    '''
    e_native_se = compute_sequences_energy_ix(exon_seq_index,model_exon.potts_model,model_exon.mask) #SE
    
    return e_native_se



def compute_e_se_config_MSA(ini,fin,decoy_seqs,mask,potts_model):
    '''
    using resnum 
    '''
    mask=np.ones(mask.shape)*mask.mean()
        
    # strictly mask the mean mask to fit this fragment 
    fragment_pos=np.arange(ini-1,fin)
    
    mask_=compute_fragment_mask_SE(mask,fragment_pos)
        
    e_decoy_se = compute_sequences_energy(decoy_seqs,potts_model,mask_,config_decoys = True,
                             WP = False,fragment_pos=fragment_pos) #SE
    
    return e_decoy_se.mean(),e_decoy_se.std()

def concatenate_MSA(MSA_,len_pre,len_post):
    if len_pre>0:
        arr_pre=np.array([['-']*MSA_.shape[0]]*len_pre)
        aux=np.concatenate([arr_pre,MSA_.T])
    else:
        aux=MSA_.T
    if len_post>0:
        arr_post=np.array([['-']*MSA_.shape[0]]*len_post)
        MSA_concat=np.concatenate([aux,arr_post]).T
    else:
        MSA_concat=aux.T
    return np.char.upper(MSA_concat)


#### real exon decoys

def make_control_fragments(beta,seq_len,min_len):
    # N: ensemble size
    n_exons=int(seq_len*beta*20) # n_exons: much more than expected for a single sequence
    simulated_lengths = np.random.geometric(p=beta, size=n_exons)
    simulated_lengths = simulated_lengths[simulated_lengths>min_len]
    bs=np.cumsum(simulated_lengths)
    bs=bs[bs>min_len]
    bs=bs[bs<(seq_len-min_len)]
    bs=np.hstack([np.zeros(1,int),bs,np.array(seq_len)])
    return bs

def get_control_fragments(sim_bs,ali_seq_num_pdb,pdb_beg):
   
    #exon_start=[ali_seq_num_pdb[final_bs[i]] for i in range(len(final_bs[:-1]))]
    #exon_end=[ali_seq_num_pdb[final_bs[i]-1] for i in range(1,len(final_bs))]

    #exons=np.array([exon_start,exon_end]).T
    clean_pdb_pos=np.append(ali_seq_num_pdb,ali_seq_num_pdb[-1]+1)+1-pdb_beg+1

    exon_rel_start=[clean_pdb_pos[sim_bs[i]] for i in range(len(sim_bs[:-1]))]
    exon_rel_end=[clean_pdb_pos[sim_bs[i]] for i in range(1,len(sim_bs))] #CHECK +1
    sim_exons_rel_pos=np.array([exon_rel_start,exon_rel_end]).T

    sim_exon_len=[(b-a+1) for a,b in sim_exons_rel_pos]

    return sim_exons_rel_pos,sim_exon_len,exon_rel_start,exon_rel_end
def make_real_exon_decoys(beta,N,seq_len,min_len,ali_seq_num_pdb,pdb_beg):
    sim_exon_len_=[]
    exon_rel_start_=[]
    exon_rel_end_=[]
    for i in range(N):
        sim_bs=make_control_fragments(beta,seq_len,min_len)
        sim_exons_rel_pos,sim_exon_len,exon_rel_start,exon_rel_end=get_control_fragments(sim_bs,
                                                                                         ali_seq_num_pdb,
                                                                                         pdb_beg)
        sim_exon_len_+=sim_exon_len
        exon_rel_start_+=exon_rel_start
        exon_rel_end_+=exon_rel_end
    selected_exons_pos=pd.DataFrame({'exon_start_pdb':exon_rel_start_,'exon_end_pdb':exon_rel_end_})
    return selected_exons_pos
    