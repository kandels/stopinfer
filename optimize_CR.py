import uproot
import pandas as pd
import numpy as np
import math
campaign = 'Run3'
lumi = 140_000 if campaign == 'Run2' else 53_000
in_dir = "/afs/cern.ch/user/s/skandel/stop_ANA-SUSY-2023-08/bWN1/CompDNN/LSFHad/"
presel = "((nlep == 2 and lep1pT > 25 and lep2pT > 20 and mll > 20 and  MDR > 75 and isOS and RPT > 0.70 and METsig > 10) and (isSignalLep1 == 1 and isSignalLep2 == 1) and (isSF and (mll < 71.2 or mll > 111.2) or ~isSF))"
weights = ["xsec", "WeightLumi", "WeightEventsPU", "WeightEvents", "WeightEventselSF", "WeightEventsmuSF", "WeightEventsJVT", "WeightEventsbTag", "WeightEventsSF_global"]
dnn = ["DNN_smallDM_bjets0_sig", "DNN_smallDM_bjets0_ttbar", "DNN_smallDM_bjets0_VV",  "DNN_smallDM_bjets1_sig", "DNN_smallDM_bjets1_ttbar", "DNN_smallDM_bjets1_VV", "DNN_largeDM_bjets0_sig", "DNN_largeDM_bjets0_ttbar", "DNN_largeDM_bjets0_VV", "DNN_largeDM_bjets1_sig", "DNN_largeDM_bjets1_ttbar", "DNN_largeDM_bjets1_VV", "RPT", "nbjet", "METsig"]
other = ["lep1pT", "lep2pT", "mll", "nlep", "isOS", "isSF", "isSignalLep1", "isSignalLep2", "MDR"] #nbjet METsig
out = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/DNN/SROptStudy/nbjetmodel/METsig/" #RPT METsig
reg_cut = {'smalldm0b': '(nbjet==0) and (METsig >10) and (RPT > 0.70)', 'smalldm1b': '(nbjet>0) and (METsig >10) and (RPT > 0.70)', 'largedm0b':'(nbjet==0) and (METsig >10) and (RPT > 0.70)', 'largedm1b':'(nbjet>0) and (METsig >10) and (RPT > 0.70)', 'SRSF1b': '(isSF and (mll < 71.2 or mll > 111.2) and nbjet >0)', 'SRSF0b': '(isSF and (mll < 71.2 or mll > 111.2) and nbjet==0)', 'SRDF1b': '(~isSF and nbjet>0)', 'SRDF0b': '(~isSF and nbjet==0)'}
do_csv = False


def load_bkg(bkg: str):
    import glob
    w_card = 'Bkg*mc20*.root' if campaign == 'Run2' else 'Bkg*mc23*.root'
    files = glob.glob(in_dir + w_card)
    tree_name = bkg + "_WEIGHTS"
    file_tree_map = {f: tree_name for f in files}
    print(f"Found files: {files}")
    f = uproot.concatenate(file_tree_map, weights + dnn + other, library='pd')
    f = f.query(presel)
    print(f.shape)
    f = f.drop(columns=other)
    print(f.shape)
    f.to_csv(out + bkg + "_" + campaign + ".csv", index=False)
    return


def load_sig(sig:str):
    import glob
    w_card = 'Sig*mc20*.root' if campaign == 'Run2' else 'Sig*mc23*.root'
    files = glob.glob(in_dir + w_card)
    tree_name = sig + "_WEIGHTS"
    file_tree_map = {f: tree_name for f in files}
    f = uproot.concatenate(file_tree_map, weights + dnn + other, library='pd')
    f = f.query(presel)
    print(f.shape)
    f = f.drop(columns=other)
    print(f.shape)
    f.to_csv(out + sig + "_" + campaign + ".csv", index=False)
    return


if do_csv:
    bkgs = ['ttbar', 'VV', 'Zjets_Sh', 'Wt_dyn_DR', 'other', 'ttZ']
    for b in bkgs:
        print(b)
        load_bkg(b)
        
    benchmark = ["TT_bWN1_550_460", "TT_bWN1_550_385", "TT_bWN1_650_485", "TT_bWN1_650_560"]
    for b in benchmark:
        load_sig(b)


def my_significance(s, b, sig):
    import math
    n = s + b
    sigma = b * sig
    z = 0
    if s > 0 and b > 0:
        z = math.sqrt(2 * (n * math.log(n * (b + sigma * sigma) / (b * b + n * sigma * sigma)) - b * b / (sigma * sigma) * math.log((b * b + n * sigma * sigma) / (b * (b + sigma * sigma))) ) )
	
    return z

def optimize_cr(campaign = campaign, background='ttbar', s_point='TT_bWN1_650_560', region='smalldm1b'):
    """
    Optimize the CR for ttbar and VV.
    CR to be designed
        ttbar
            -Small1bRun2
            -Large1bRun2
        VV
            -Small0bRun2
            -Large0bRun2
    Definition - Cut on Signal score is applied. CR - [0, left bound of VR]. VR - [upper bound of CR, lower bound of SR].
    Then to improve purity, cut on background score is applied.
    
    Strategy - Scan signal score from 0 - some bound of SR. Identiy the range where s/b < 0.05 --> CR. 0.05 < s/b < 0.1 --> VR.
    """
        
    region_scores = {
        'smalldm0b':{
            'signal': 'DNN_smallDM_bjets0_sig',
            'ttbar': 'DNN_smallDM_bjets0_ttbar',
            'VV': 'DNN_smallDM_bjets0_VV'
        },
        'smalldm1b':{
            'signal': 'DNN_smallDM_bjets1_sig',
            'ttbar': 'DNN_smallDM_bjets1_ttbar',
            'VV': 'DNN_smallDM_bjets1_VV'
        },
        'largedm0b':{
            'signal': 'DNN_largeDM_bjets0_sig',
            'ttbar': 'DNN_largeDM_bjets0_ttbar',
            'VV': 'DNN_largeDM_bjets0_VV'
        },
        'largedm1b':{
            'signal': 'DNN_largeDM_bjets1_sig',
            'ttbar': 'DNN_largeDM_bjets1_ttbar',
            'VV': 'DNN_largeDM_bjets1_VV'
        }
    }
    sr_low_bins = {
        'smalldm0bRun2': 0.60,
        'smalldm1bRun2': 0.952,
        'largedm0bRun2': 0.82,
        'largedm1bRun2': 0.85,
        
        'smalldm0bRun3': 0.748,
        'smalldm1bRun3': 0.939,
        'largedm0bRun3': 0.856,
        'largedm1bRun3': 0.897,
    }
    
    score_sig = region_scores[region]['signal']
    score_ttbar = region_scores[region]['ttbar']
    score_VV = region_scores[region]['VV']
    
    bkg=['ttbar', 'VV', 'Zjets_Sh', 'Wt_dyn_DR', 'other', 'ttZ']
    bkg_dfs = []
    for b in bkg:
        b_df = pd.read_csv(out + b + "_" + campaign + ".csv", index_col=False)
        b_df = b_df.query(reg_cut[region]).copy()
        b_df['yield'] = lumi * b_df['xsec'] * b_df['WeightLumi'] * b_df['WeightEventsPU'] * b_df['WeightEvents'] * b_df['WeightEventselSF'] * b_df['WeightEventsmuSF'] * b_df['WeightEventsJVT'] * b_df['WeightEventsbTag'] * b_df['WeightEventsSF_global']
        bkg_dfs.append(b_df)
    mc = pd.concat(bkg_dfs)
    
    sig_df = pd.read_csv(out + s_point + "_" + campaign + ".csv", index_col=False)
    sig_df = sig_df.query(reg_cut[region]).copy()
    sig_df['yield'] = lumi * sig_df['xsec'] * sig_df['WeightLumi'] * sig_df['WeightEventsPU'] * sig_df['WeightEvents'] * sig_df['WeightEventselSF'] * sig_df['WeightEventsmuSF'] * sig_df['WeightEventsJVT'] * sig_df['WeightEventsbTag'] * sig_df['WeightEventsSF_global']
    
    print('Sanity Check SIG preselection Yield' ,sig_df['yield'].sum())
    print('Sanity Check BKG preselection Yield' ,mc['yield'].sum())
    
    CR_UPPER_BOUND = min(sr_low_bins[f'{region}{campaign}'], 0.4)
    # now find vr lower bound - this is where 0.05 < s/b < 0.1
    print("=" * 50)
    print(f'Starting VR lower bin search - {campaign} in region {region} for background {background}')
    print("=" * 50)
    
    step = 0.01
    scan_range = np.arange(CR_UPPER_BOUND-step, 0, -step)
    cr_lower_bound = None
    last_valid_bound = None
    found_cr_range = False
    
    for upper_bound in scan_range:
        s = sig_df[(sig_df[score_sig] >= 0) & (sig_df[score_sig] < upper_bound)]['yield'].sum()
        b = mc[(mc[score_sig] >= 0) & (mc[score_sig] < upper_bound)]['yield'].sum()
       
        if b == 0:
            s_b_ratio = np.nan
        else:
            s_b_ratio = s/b

        print(f'In bin [{0:.3f}, {upper_bound:.3f}] sig = {s:.3f} bkg = {b:.3f} s/b = {s_b_ratio:.4f}')
        
        # Check if this bin is in the CR range
        if 0.01 < s_b_ratio:
            last_valid_bound = upper_bound  # Keep track of the last valid VR bin
            found_cr_range = True
        elif found_cr_range:
            # We've found CR range before and now we're outside it
            cr_lower_bound = last_valid_bound
            print('-' * 50)
            print(f'CR range ended. Last valid CR bin was at {last_valid_bound:.3f}')
            print(f'Current bin [{upper_bound:.3f}, {CR_UPPER_BOUND:.3f}] has s/b = {s_b_ratio:.4f} (outside VR range)')
            break
        # If we haven't found CR range yet, continue scanning
    
    if cr_lower_bound is None:
        if found_cr_range:
            cr_lower_bound = last_valid_bound
        else:
            print("No CR range found - all bins have s/b outside 0.05-0.1 range")
            cr_lower_bound = "Not found"
        
    print('=' * 50)
    print(f'CR Signal score bounds for {campaign} {background} in region {region} are - [{cr_lower_bound}, {CR_UPPER_BOUND}]')
    print('=' * 50)
        
    return


def purity_contamination(campaign, background, s_point, region):
    """
    Increase purity and reduce signal contamination of background in the control region. 
    Purity = no of background for which CR is designed / total number of background events.
    Contamination = number of signal / total events(bkg + sig).
    """
    
    control_regions = {
        'ttbar': {
            'smalldm0brun2': [], 
            'largedm0brun2': [],
            'smalldm1brun2': [],
            'largedm1brun2': [],
            'smalldm0brun3': [0.248, 0.748],
            'largedm0brun3': [0.450, 0.856],
            'smalldm1brun3': [0.899, 0.939],
            'largedm1brun3': [0.487, 0.897],
        },
        'VV':{
            'smalldm0brun2': [],
            'largedm0brun2': [],
            'smalldm0brun3': [],
            'largedm0brun3': [],
        }
    }
    return

optimize_cr(campaign=campaign, background='VV', s_point='TT_bWN1_550_385', region='largedm0b')
    