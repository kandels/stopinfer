import uproot
import pandas as pd
import numpy as np
import math
campaign = 'Run3'
lumi = 140_000 if campaign == 'Run2' else 53_000
in_dir = "/afs/cern.ch/user/s/skandel/stop_ANA-SUSY-2023-08/bWN1/CompDNN/LSFHad/"
#presel = "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 75 && isOS) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (isSF && (mll < 71.2 || mll > 111.2) || !isSF) && (RPT > 0.70))"
#RPT
#presel = "((nlep == 2 and lep1pT > 25 and lep2pT > 20 and mll > 20 and  MDR > 75 and isOS and RPT > 0.70) and (isSignalLep1 == 1 and isSignalLep2 == 1) and (isSF and (mll < 71.2 or mll > 111.2) or ~isSF) and (MDR > 75))"
#RPT METsig
presel = "((nlep == 2 and lep1pT > 25 and lep2pT > 20 and mll > 20 and  MDR > 75 and isOS and RPT > 0.70 and METsig > 10) and (isSignalLep1 == 1 and isSignalLep2 == 1) and (isSF and (mll < 71.2 or mll > 111.2) or ~isSF))"
#SFDF
#presel = "((nlep == 2 and lep1pT > 25 and lep2pT > 20 and mll > 20 and  MDR > 75 and isOS and METsig > 10) and (isSignalLep1 == 1 and isSignalLep2 == 1) and (isSF and (mll < 71.2 or mll > 111.2) or ~isSF) and (MDR > 75))"

weights = ["xsec", "WeightLumi", "WeightEventsPU", "WeightEvents", "WeightEventselSF", "WeightEventsmuSF", "WeightEventsJVT", "WeightEventsbTag", "WeightEventsSF_global"]
#bjet model
dnn = ["DNN_smallDM_bjets0_sig", "DNN_smallDM_bjets0_ttbar", "DNN_smallDM_bjets0_VV",  "DNN_smallDM_bjets1_sig", "DNN_smallDM_bjets1_ttbar", "DNN_smallDM_bjets1_VV", "DNN_largeDM_bjets0_sig", "DNN_largeDM_bjets0_ttbar", "DNN_largeDM_bjets0_VV", "DNN_largeDM_bjets1_sig", "DNN_largeDM_bjets1_ttbar", "DNN_largeDM_bjets1_VV", "RPT", "nbjet", "METsig"]

#sfdf model
#dnn = ["DNN_SF0b_sig", "DNN_SF1b_sig", "DNN_DF0b_sig", "DNN_DF1b_sig", "DNN_SF0b_ttbar", "DNN_SF1b_ttbar", "DNN_DF0b_ttbar", "DNN_DF1b_ttbar", "DNN_SF0b_VV", "DNN_DF0b_VV", "METsig", "nbjet", "isSF", "mll"]

#other = ["lep1pT", "lep2pT", "mll", "nlep", "isOS", "isSF", "isSignalLep1", "isSignalLep2", "MDR"] #nbjet
other = ["lep1pT", "lep2pT", "mll", "nlep", "isOS", "isSF", "isSignalLep1", "isSignalLep2", "MDR"] #nbjet METsig

#out = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/DNN/SROptStudy/nbjetmodel/" #RPT
out = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/DNN/SROptStudy/nbjetmodel/METsig/" #RPT METsig

#out = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/DNN/SROptStudy/metsig10sfdfmodel/" #sfdf
#other = ["lep1pT", "lep2pT", "nlep", "isOS", "isSignalLep1", "isSignalLep2", "MDR"] #sfdf
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

def optimize_sr_by_sb_threshold(s_point, region='smallDM0b', bkg=['ttbar', 'VV', 'Zjets_Sh', 'Wt_dyn_DR', 'other', 'ttZ'], out=out, high=1, flat_sys=0.2, step=0.001, min_events=0, min_bkg_events=2, sr_min_sb_ratio=0.1, campaign=campaign):
    # Strategy - Sequentially find SR bins with s/b >= sr_min_sb_ratio, stop when s/b falls below
    bkg_dfs = []
    reg_score = {'smallDM0b': 'DNN_smallDM_bjets0_sig', 'smallDM1b': 'DNN_smallDM_bjets1_sig', 'largeDM0b': 'DNN_largeDM_bjets0_sig', 'largeDM1b': 'DNN_largeDM_bjets1_sig', 'SRSF0b':'DNN_SF0b_sig', 'SRSF1b':'DNN_SF1b_sig', 'SRDF0b':'DNN_DF0b_sig', 'SRDF1b':'DNN_SF1b_sig'}
    reg_cut = {'smallDM0b': 'nbjet==0', 'smallDM1b': 'nbjet>0', 'largeDM0b':'nbjet==0', 'largeDM1b':'nbjet>0', 'SRSF1b': '(isSF and (mll < 71.2 or mll > 111.2) and nbjet >0)', 'SRSF0b': '(isSF and (mll < 71.2 or mll > 111.2) and nbjet==0)', 'SRDF1b': '(~isSF and nbjet>0)', 'SRDF0b': '(~isSF and nbjet==0)'}
    opt_score = reg_score[region]
    print('Using cut', reg_cut[region])
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

    sr_bin_edges = [high]
    current_upper_bound = high

    #print("\nOptimizing Signal Region (s/b >= {:.2f}) with signal point {}:".format(sr_min_sb_ratio))
    print(f"Campaing {campaign} Optimizing SR {region} with benchmark point {s_point} s > {min_events} b >= {min_bkg_events} s/b > {sr_min_sb_ratio}:")
    while True:
        best_significance = -1.0
        optimal_lower_bound = None
        found_valid_bin = False

        scan_dnn_value = np.arange(current_upper_bound - step, 0, -step) # Scan down to 0

        if not scan_dnn_value.size > 0:
            break

        for lower_bound in scan_dnn_value:
            s_bin = sig_df[(sig_df[opt_score] >= lower_bound) & (sig_df[opt_score] < current_upper_bound)]['yield'].sum()
            b_bin = mc[(mc[opt_score] >= lower_bound) & (mc[opt_score] < current_upper_bound)]['yield'].sum()
            b_sigma = flat_sys

            if s_bin >= min_events and b_bin >= min_bkg_events and (s_bin / b_bin) >= sr_min_sb_ratio:
                significance = my_significance(s_bin, b_bin, b_sigma)
                if significance > best_significance:
                    best_significance = significance
                    optimal_lower_bound = lower_bound
                    print(f'  SR Bin: [{optimal_lower_bound:.4f}, {current_upper_bound:.4f}] s: {s_bin:.2f} b: {b_bin:.2f} s/b: {s_bin/b_bin:.3f} signif: {best_significance:.3f}')
                    found_valid_bin = True

        if optimal_lower_bound is not None:
            sr_bin_edges.append(optimal_lower_bound)
            current_upper_bound = optimal_lower_bound
        elif not found_valid_bin:
            break # Stop SR optimization if no valid bin found in this iteration

    sr_bin_edges.sort(reverse=True)
    sr_lower_limit = min(sr_bin_edges) if sr_bin_edges else 0.0

    # --- Plotting ---
    if len(sr_bin_edges) > 1:
        bin_edges = sorted(sr_bin_edges, reverse=True) # Ensure high to low
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
        signal_yields = [sig_df[(sig_df[opt_score] >= bin_edges[i+1]) & (sig_df[opt_score] < bin_edges[i])]['yield'].sum() for i in range(len(bin_edges) - 1)]
        bkg_yields = [mc[(mc[opt_score] >= bin_edges[i+1]) & (mc[opt_score] < bin_edges[i])]['yield'].sum() for i in range(len(bin_edges) - 1)]
        bin_labels = [f'{bin_edges[i+1]:.2f}-{bin_edges[i]:.2f}' for i in range(len(bin_edges) - 1)]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        width = 0.35
        x = np.arange(len(bin_centers))

        rects1 = plt.bar(x - width/2, signal_yields, width, label=f'Signal {s_point}')
        rects2 = plt.bar(x + width/2, bkg_yields, width, label='Background')

        plt.ylabel('Yield')
        plt.xlabel('DNN Score Bin')
        plt.title(f'Signal and Background Yields in SR - {region}')
        plt.xticks(x, bin_labels, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'optimal_{region}_{s_point}.pdf')
        #plt.show()
    else:
        print("\nNo valid SR bins found based on the s/b threshold.")

    print("\nSignal Region Lower Limit (based on s/b < {:.2f} below it): {:.4f}".format(sr_min_sb_ratio, sr_lower_limit))
    return {"sr_bins": sr_bin_edges, "sr_lower_limit": sr_lower_limit}

optimal_sr = optimize_sr_by_sb_threshold(s_point="TT_bWN1_650_485", region='largeDM1b', out=out, high=1.0, step=0.001, min_events=0, sr_min_sb_ratio=0.1, campaign=campaign)

if optimal_sr:
    print("\nFinal Optimized Signal Region Bins (High to Low DNN):", optimal_sr["sr_bins"])
    print("Signal Region Lower Limit:", optimal_sr["sr_lower_limit"])