import ROOT
process = ["VV_WEIGHTS", "Wt_dyn_DR_WEIGHTS", "ttZ_WEIGHTS", "other_WEIGHTS", "ttbar_WEIGHTS", "MCFakes_WEIGHTS", "Zjets_Sh_WEIGHTS"]
#process = ["MCFakes_WEIGHTS"]
#process = ["Zjets_Sh_WEIGHTS"]
campaign = ["mc20a", "mc20d", "mc20e", "mc23a", "mc23d"]
#preselection = "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 75 && isOS && RPT >0.70) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (isSF && (mll < 71.2 || mll > 111.2) || !isSF))"
#preselection = "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 75 && isOS) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (isSF && (mll < 71.2 || mll > 111.2) || !isSF) && (RPT > 0.70))"
#preselection = "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 75 && isOS && RPT > 0.70) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (isSF && (mll < 71.2 || mll > 111.2) || !isSF)&&(MDR > 75))"
#preselection = "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 75 && isOS && METsig > 10) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (isSF && (mll < 71.2 || mll > 111.2) || !isSF))"
#preselection = "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 75 && isOS) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (isSF && (mll < 71.2 || mll > 111.2) || !isSF))"
preselection = "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 75 && isOS && RPT > 0.70 && METsig>10) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (isSF && (mll < 71.2 || mll > 111.2) || !isSF))"

for p in process:
    for c in campaign:
        chain = ROOT.TChain(p)
        chain.Add(f"full/Bkgstop3b.{c}.WEIGHTS.root")
        rdf = ROOT.RDataFrame(chain)
        rdf = rdf.Filter(preselection)
        print(p, c, "Entries", rdf.Count().GetValue())
        rdf.Snapshot(p, p+"_"+c+".root")
