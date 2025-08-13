from configManager import configMgr
from ROOT import kBlack, kWhite, kGray, kRed, kPink, kMagenta, kViolet, kBlue, kAzure, kCyan, kTeal, kGreen, kSpring, kYellow, kOrange, kDashed, kSolid, kDotted
from configWriter import fitConfig, Measurement, Channel, Sample
from math import sqrt
from systematic import Systematic
from copy import deepcopy
from optparse import OptionParser
from logger import Logger
import ROOT
import os
import sys
import glob
import json

from ROOT import gROOT, TLegend, TLegendEntry, TCanvas

from cutList import cutList
from SetSystematic import SetSystematic

# add sys path later

def Lumi(period = "Run2"):
    lumi = {}
    lumi["2015"] = 3.24454;
    lumi["2016"] = 33.4022;
    lumi["2017"] = 44.6306;
    lumi["2018"] = 58.7916;
    lumi["2022"] = 26.0714;
    lumi["2023"] = 27.2399;
    lumi["Run2"] = (lumi["2015"] + lumi["2016"] + lumi["2017"] + lumi["2018"])
    lumi["Run3"] = (lumi["2022"] + lumi["2023"])
    return float(lumi[period])

myfolder = os.system("pwd")

cList3b = cutList('Stop3b')
configMgr.cutsDict = cList3b.cutsDict

#Load Options Parser.
myUserArgs = configMgr.userArg.split(' ')
myInputParser = OptionParser()
myInputParser.add_option('--period', dest='period', choices=("Run2","Run3","201516","2017","2018","2022","2023"), default='Run2')
myInputParser.add_option("--signal", dest='signal', default='TT_bWN1_550_460', help="Signal Model 1")
myInputParser.add_option("--doSignalPlots", dest='doSignalPlots', action='store_true', default=False, help="do signal plots")
myInputParser.add_option("--doPlots"      , dest='doPlots'      , action='store_true', default=False, help="do plots")
myInputParser.add_option("--doSyst"       , dest='doSyst',              action='store_true',                   default=False, help="doSyst")
myInputParser.add_option("--doTheoSyst"   , dest='doTheoSyst',          action='store_true',                   default=False,  help="doTheoSyst")
myInputParser.add_option("--region"       , dest='region',              choices=("small","large"),       default='small')
myInputParser.add_option('--createRegionHist', dest='createRegionHist', default='')
myInputParser.add_option('--createSampleHist', dest='createSampleHist', default='')
myInputParser.add_option('--regiontoplot'       , dest='regiontoplot'   , default='CRTOPsmall1bRun2')
myInputParser.add_option('--variabletoplot'     , dest='variabletoplot' , default='MET')

(options, args) = myInputParser.parse_args(myUserArgs)
# FitType comes from here - https://github.com/histfitter/histfitter/blob/4274c73c71d626c5752280e1ea637c2cf0cf81d9/scripts/HistFitter.py#L123
if options.doSyst or (myFitType == FitType.Exclusion) or (myFitType == FitType.Discovery):
    ROOT.gErrorIgnoreLevel = ROOT.kFatal

Control_region_name      = [f"CRVV{options.region}0b{options.period}",f"CRTOP{options.region}1b{options.period}"]
if options.period == 'Run2':    
    if options.region=='small':
        SignalRegion_region_name = [f"SR{options.period}{options.region}DM0b0", f"SR{options.period}{options.region}DM1b0", f"SR{options.period}{options.region}DM0b1", f"SR{options.period}{options.region}DM1b1", f"SR{options.period}{options.region}DM0b2"]
    elif options.region=='large':
        SignalRegion_region_name = [f"SR{options.period}{options.region}DM0bALL", f"SR{options.period}{options.region}DM1b0", f"SR{options.period}{options.region}DM1b1"]

elif options.period == 'Run3':
    if options.region=='small':
        SignalRegion_region_name = [f"SR{options.period}{options.region}DM0b0", f"SR{options.period}{options.region}DM1bALL", f"SR{options.period}{options.region}DM0b1", f"SR{options.period}{options.region}DM0b2"]
    elif options.region=='large':
        SignalRegion_region_name = [f"SR{options.period}{options.region}DM0b0", f"SR{options.period}{options.region}DM1b0", f"SR{options.period}{options.region}DM0b1", f"SR{options.period}{options.region}DM1b1"]

else:
    SignalRegion_region_name = [f"SRNN{options.region}1b0{options.period}",f"SRNN{options.region}2b0{options.period}",f"SRNN{options.region}1b1{options.period}",f"SRNN{options.region}2b1{options.period}",f"SRNN{options.region}1b2{options.period}",f"SRNN{options.region}2b2{options.period}",f"SRNN{options.region}1b3{options.period}",f"SRNN{options.region}2b3{options.period}"]
ValidationRegion_region_name = [f"VRVV{options.region}0b{options.period}",f"VRTOP{options.region}1b{options.period}"] 

# ---------------------------------------
print("Control regions: ", Control_region_name)
# ---------------------------------------
print("Signal regions: ", SignalRegion_region_name)
# ---------------------------------------

for region in Control_region_name:
        if cList3b.IsoSig not in configMgr.cutsDict[region]:
                print(configMgr.cutsDict[region])
                print("IsoSig not defined for region", region )
                exit(1)
for region in SignalRegion_region_name:
        if cList3b.IsoSig not in configMgr.cutsDict[region]:
                print(configMgr.cutsDict[region])
                print("IsoSig not defined for region", region )
                exit(1)
for region in ValidationRegion_region_name:
        if cList3b.IsoSig not in configMgr.cutsDict[region]:
                print(configMgr.cutsDict[region])
                print("IsoSig not defined for region", region )
                exit(1)
if options.createRegionHist!= '': print(options.createRegionHist)
if options.createSampleHist!= '': print(options.createSampleHist)

# ---------------------------------o------
# Flags to control which fit is executed
# ---------------------------------------
useStat = True
configMgr.inputLumi = 1.
configMgr.outputLumi = Lumi(options.period)
print(configMgr.outputLumi)
configMgr.setLumiUnits("pb-1")
configMgr.blindSR = True
configMgr.blindCR = False
configMgr.blindVR = False

configMgr.scanRange = (0, 1)

# configMgr.doHypoTest=True
configMgr.nTOYs = 1000
configMgr.calculatorType = 2  # 0 for toys
configMgr.testStatType = 3
configMgr.nPoints = 50
configMgr.ReduceCorrMatrix = True  # Boolean to make a reduced correlation matrix
if options.signal == "ttHinv":    
    configMgr.scanRange = (0, 1.5)

configMgr.analysisName = "MyFit_3body_"+options.region+"_"+options.period
if options.doTheoSyst and options.doSyst == False:
    configMgr.analysisName += "_theoSyst"
if myFitType == FitType.Discovery:
    configMgr.analysisName = "Disc_syst_"+ options.region + "_" +options.period+ "_" +options.signal
if myFitType == FitType.Exclusion:
    configMgr.analysisName = "Excl_syst_"+ options.region + "_" +options.period+ "_" +options.signal
if options.createRegionHist != '' and options.createSampleHist != '':
    configMgr.analysisName = "ForMyCachefile_v1_" + options.createRegionHist+"_"+options.createSampleHist
if options.doPlots:
    configMgr.analysisName = "DistrosStop3body_"+options.period+"_"+options.regiontoplot+"_"+options.variabletoplot
    if options.region not in options.regiontoplot:
        print("Region not in region to plot")
        exit(1)
if options.doSignalPlots:
    configMgr.analysisName = "SignalPlotsStop3body_"+options.region+"_"+options.period


# fix these cache later
configMgr.histCacheFile = "data/"+configMgr.analysisName+"_Cache.root"
configMgr.histBackupCacheFile = f"cachefile/FullAug5/MyFit_3body_{options.region}_{options.period}_Cache.root"

if options.doTheoSyst and options.doSyst:
    # configMgr.histBackupCacheFile = f"cachefile/Jul29_theo/MyFit_2Lep_{options.region}_{options.period}_theoSyst_Cache.root"
    configMgr.histBackupCacheFile = f"cachefile/FullJul29/MyFit_2Lep_{options.region}_{options.period}_Cache.root"

print("Cache file: ", configMgr.histBackupCacheFile)
configMgr.outputFileName = "results/"+configMgr.analysisName+"_Output.root"

configMgr.useHistBackupCacheFile = False
if options.doSyst:
        configMgr.useCacheToTreeFallback = False
        configMgr.forceNorm = False
else:
        configMgr.useCacheToTreeFallback = False
        configMgr.forceNorm = False

# Set the files to read from
version = "CompDNN/LSFHad/"
pathInput = "/afs/cern.ch/user/s/skandel/stop_ANA-SUSY-2023-08/bWN1/"+version

# CRITICAL: Set the nominal tree name (this was missing!)
configMgr.nomName = "_WEIGHTS"

# Force tree reading settings
configMgr.readFromTree = True
configMgr.executeHistFactory = True
configMgr.writeXML = True

ntupFiles = []
ntupFilessig = []
ntupFilesdata = []

configMgr.weights = ([
    "WeightEvents",
    "xsec",
    "WeightEventsPU",
    "WeightLumi",
    "WeightEventselSF",
    "WeightEventsmuSF",
    "WeightEventsJVT",
    "WeightEventsbTag",
    "WeightEventsSF_global",
])

print(configMgr.weights)

if options.period == 'Run2':
    ntupFilessig = [f"{pathInput}/Sigstop3b.mc20a.WEIGHTS.root", f"{pathInput}/Sigstop3b.mc20d.WEIGHTS.root", f"{pathInput}/Sigstop3b.mc20e.WEIGHTS.root"]
    ntupFilesdata = [f"{pathInput}/data15_13TeV.periodAllYear.stop3b_NONE.root", f"{pathInput}/data16_13TeV.periodAllYear.stop3b_NONE.root", f"{pathInput}/data17_13TeV.periodAllYear.stop3b_NONE.root", f"{pathInput}/data18_13TeV.periodAllYear.stop3b_NONE.root"]
    ntupFiles = [f"{pathInput}/Bkgstop3b.mc20a.WEIGHTS.root", f"{pathInput}/Bkgstop3b.mc20d.WEIGHTS.root", f"{pathInput}/Bkgstop3b.mc20e.WEIGHTS.root"]

elif options.period == 'Run3':
    ntupFilessig = [f"{pathInput}/Sigstop3b.mc23a.WEIGHTS.root", f"{pathInput}/Sigstop3b.mc23d.WEIGHTS.root"]
    ntupFilesdata = [f"{pathInput}/data22_13p6TeV.periodAllYear.stop3b_NONE.root", f"{pathInput}/data23_13p6TeV.periodAllYear.stop3b_NONE.root"]
    ntupFiles = [f"{pathInput}/Bkgstop3b.mc23a.WEIGHTS.root", f"{pathInput}/Bkgstop3b.mc23d.WEIGHTS.root"]
else:
    print('Invalid period')
    exit(1)

# DEBUG: Print critical settings
print("=== DEBUG INFO ===")
print("configMgr.nomName:", configMgr.nomName)
print("configMgr.readFromTree:", configMgr.readFromTree)
print("Background input files:", ntupFiles)
print("Data input files:", ntupFilesdata)
print("=== END DEBUG ===")

# skipping the systematics part now
syst = SetSystematic()
syst.DataTreeName = "data_NONE"
syst.systNONE = configMgr.nomName
syst.cutsList = cList3b

top = Sample("ttbar", kYellow)
if options.doTheoSyst:
        top.suffixTreeName = "_THEORY"
top.setStatConfig(useStat)
top.addSampleSpecificWeight(cList3b.isPrompt)
top.addSampleSpecificWeight("(nlep==2)")
if options.createRegionHist == '' and options.createSampleHist == '':
        top.setNormRegions([(f"CRTOP{options.region}1b{options.period}", "cuts")])
        top.setNormFactor(f"mu_top_{options.period}", 1., 0., 5.)

if top.name == "ttbar_Sh":
        configMgr.histCacheFile = configMgr.histCacheFile.replace(configMgr.analysisName,configMgr.analysisName+"_Sh")
        configMgr.outputFileName = configMgr.outputFileName.replace(configMgr.analysisName,configMgr.analysisName+"_Sh")
        configMgr.analysisName = configMgr.analysisName + "_Sh"


Diboson = Sample("VV", kMagenta+2)
Diboson.addSampleSpecificWeight(cList3b.isPrompt)
Diboson.setNormByTheory()
Diboson.setStatConfig(useStat)
if options.createRegionHist == '' and options.createSampleHist == '':
        Diboson.setNormRegions([(f"CRVV{options.region}0b{options.period}", "cuts")])
        Diboson.setNormFactor(f"mu_vv_{options.period}", 1., 0., 5.)


ttZ = Sample("ttZ", kBlue)
ttZ.addSampleSpecificWeight(cList3b.isPrompt)
ttZ.setStatConfig(useStat)

print("suffixTreeName: ", ttZ.suffixTreeName)

singletop = Sample("Wt_dyn_DR", kRed)
singletop.addSampleSpecificWeight(cList3b.isPrompt)
singletop.addSampleSpecificWeight("(nlep==2)")
singletop.setNormByTheory()
singletop.setStatConfig(useStat)

Zjets = Sample("Zjets_Sh", kSpring)
Zjets.addSampleSpecificWeight(cList3b.isPrompt)
Zjets.addSampleSpecificWeight("(nlep==2)")
Zjets.setNormByTheory()
Zjets.setStatConfig(useStat)

others = Sample("other", kGreen+4)
others.addSampleSpecificWeight(cList3b.isPrompt)
others.setNormByTheory()
others.setStatConfig(useStat)

MCFakes = Sample("MCFakes", kOrange)
MCFakes.addSampleSpecificWeight(cList3b.isNOPrompt)
# MCFakes.addSampleSpecificWeight("("+cList3b.removeSherpa+")") # fix it later
MCFakes.setNormByTheory()
MCFakes.setStatConfig(useStat)

if options.doSyst:
    # Temporary #
    samples = ['ttbar', 'VV', 'ttZ', 'Wt', 'Zjets', 'others', 'MCFakes']
    for ind, sam in enumerate([top, Diboson, ttZ, singletop, Zjets, others, MCFakes]):
        flatSyst = Systematic(
            name=f"BkgNorm_{samples[ind]}",
            nominal=1,
            high=1.2,
            low=0.8,
            type="user",
            method="userOverallSys",
        )
        sam.addSystematic(flatSyst)


data = Sample("data_NONE", kBlack)
data.prefixTreeName = "data_NONE"  
data.setStatConfig(True)
data.setData()

# -------------------------------------------------
# configMgr
# -------------------------------------------------

if myFitType == FitType.Discovery:   bkt = configMgr.addFitConfig("DiscOnly")
elif myFitType == FitType.Exclusion: bkt = configMgr.addFitConfig("ExclOnly")
else:                                bkt = configMgr.addFitConfig("BkgOnly")


# -------------------------------------------------
# Signal
# -------------------------------------------------

if (myFitType == FitType.Exclusion) or (myFitType == FitType.Discovery):
    sigSample = Sample(options.signal, kPink)
    sigSample.prefixTreeName = options.signal
    sigSample.addInputs(ntupFilessig)
    sigSample.suffixTreeName = "_WEIGHTS"
    sigSample.setNormFactor("mu_SIG", 1., 0., 10.)
    sigSample.setStatConfig(useStat)
    sigSample.addSampleSpecificWeight(cList3b.isPrompt)
    bkt.setSignalSample(sigSample)
    # syst.setDetectorSyst(sigSample, configMgr.weights, True)
    # syst.setTheoreticalSyst(sigSample, configMgr.weights, True)

if options.doSignalPlots:
        print("DOING SIGNALS!")
        if options.period == "Run2":
                ntupFilessig.append(pathInput + "sig20a.root")
                ntupFilessig.append(pathInput + "sig20d.root")
                ntupFilessig.append(pathInput + "sig20e.root")
        if options.period == "Run3":
                ntupFilessig.append(pathInput + "sig23a.root")
                ntupFilessig.append(pathInput + "sig23d.root")

        sigSample1 = Sample("TT_bWN1_550_385", kPink)
        sigSample1.addInputs(ntupFilessig)
        sigSample1.addSampleSpecificWeight(cList3b.isPrompt)
        sigSample1.setNormByTheory()
        sigSample1.setStatConfig(useStat)
        sigSample1.suffixTreeName = "_WEIGHTS"

        sigSample2 = Sample("TT_bWN1_550_400", kPink)
        sigSample2.addInputs(ntupFilessig)
        sigSample2.addSampleSpecificWeight(cList3b.isPrompt)
        sigSample2.setNormByTheory()
        sigSample2.setStatConfig(useStat)
        sigSample2.suffixTreeName = "_WEIGHTS"

        sigSample3 = Sample("TT_bWN1_550_430", kPink)
        sigSample3.addInputs(ntupFilessig)
        sigSample3.addSampleSpecificWeight(cList3b.isPrompt)
        sigSample3.setNormByTheory()
        sigSample3.setStatConfig(useStat)
        sigSample3.suffixTreeName = "_WEIGHTS"

        sigSample4 = Sample("TT_bWN1_550_460", kPink)
        sigSample4.addInputs(ntupFilessig)
        sigSample4.addSampleSpecificWeight(cList3b.isPrompt)
        sigSample4.setNormByTheory()
        sigSample4.setStatConfig(useStat)
        sigSample4.suffixTreeName = "_WEIGHTS"


if options.doSignalPlots or options.doPlots:
    d = {}
    rangeplot = {}
    rangeplot["isSF"]          = [2 , -0.5, 1.5]
    rangeplot["nbjet"]         = [6 , -0.5, 5.5]
    rangeplot["njet"]          = [6 , -0.5, 5.5]
    rangeplot["lep1pT"]        = [50, 0.  , 500.]
    rangeplot["lep2pT"]        = [50, 0.  , 500.]
    rangeplot["pt_jet1"]       = [15,0.   ,300.]
    rangeplot["pt_bjet1"]      = [15,0.   ,300.]
    rangeplot["pbll"]          = [30, 0.  , 210]
    rangeplot["mu"]            = [20,0.   ,100]
    rangeplot["nVx"]           = [10,0.   ,50]
    rangeplot["MT2"]           = [40, 50., 250]
    rangeplot["MT2_minl1l2"]   = [80, 0., 400]
    rangeplot["mll"]           = [50, 0.  , 500]
    rangeplot["DPhib"]         = [32, 0.  , 3.2]
    rangeplot["dPhil1l2"]      = [32, 0.  , 3.2]
    rangeplot["MT24lep"]       = [20, 0. , 200]
    rangeplot["ptll_noZl1Zl2"] = [30, 0.  , 600]
    rangeplot["dPhiZl1Zl2"]    = [32, 0.  , 3.2]
    rangeplot["ptll_Zl1Zl2"]   = [30, 0.  , 600]
    rangeplot["mbl"]           = [50,0.   ,500]
    rangeplot["MET"]           = [15, 50 , 350.]
    rangeplot["METsig"]        = [20, 0 , 20.]
    rangeplot["RPT"]           = [20, 0   , 1]
    rangeplot["gamInvRp1"]     = [20, 0   , 1]
    rangeplot["MDR"]           = [40, 0   , 400]
    rangeplot["DPB_vSS"]       = [32, 0.  , 3.2]
    rangeplot["cosTheta_b"]    = [20, 0   , 1]
    rangeplot["ptl1l2"]        = [60, 0   , 300]
    rangeplot["costhetall"]    = [5 ,0.   ,1.]
    rangeplot["GenFiltMET"]    = [50,0    ,500000.]
    rangeplot["GenFiltHT"]     = [50,0    ,500000.]
    rangeplot["METphi"]        = [32, -3.2, 3.2]
    rangeplot["MET_corr"]      = [50, 0   , 500]
    rangeplot["MET_corrphi"]   = [32, -3.2, 3.2]
    rangeplot["closest_mll"]   = [30, 50  , 200]
    rangeplot["closest_ptll"]  = [30, 50  , 200]
    rangeplot["lep3pT"]        = [30, 0.  , 300.]
    rangeplot["mT_corr"]       = [20,0.   ,500]
    rangeplot["lep4pT"]        = [30,0.   ,300.]
    rangeplot["lep3pT"]        = [30,0.   ,300.]
    rangeplot["closest_mll"]   = [24, 30  , 150.]
    rangeplot["DNN_smallDM_bjets0_VV"] = [25, 0, 1]
    rangeplot["DNN_largeDM_bjets0_VV"] = [25, 0, 1]
    rangeplot["DNN_smallDM_bjets1_sig"] = [10, 0, 1]
    rangeplot["DNN_smallDM_bjets1_ttbar"] = [25, 0, 1]
    rangeplot["DNN_largeDM_bjets1_ttbar"] = [25, 0, 1]
    rangeplot["DNN_smallDM_bjets0_ttbar"] = [25, 0, 1] # just for validtion
    rangeplot["DNN_largeDM_bjets0_ttbar"] = [25, 0, 1] # just for validation


    nameplot = options.variabletoplot + "_" + options.regiontoplot
    print("Creating plot for ", nameplot)
    regionplot= [options.regiontoplot]
    d[nameplot] = bkt.addChannel(options.variabletoplot,regionplot, rangeplot[options.variabletoplot][0], rangeplot[options.variabletoplot][1], rangeplot[options.variabletoplot][2])
    d[nameplot].useOverflowBin = True
    bkt.addValidationChannels(d[nameplot])


# -------------------------------------------------
# Add Input to Samples
# -------------------------------------------------

for sam in [data]:
    sam.addInputs(ntupFilesdata)

for sam in [top, singletop, ttZ, Diboson, MCFakes, Zjets, others]:
    sam.addInputs(ntupFiles)
if options.createRegionHist == '' and options.createSampleHist == '':
        if options.doPlots:
                # For plotting, exclude signal samples to show only backgrounds and data
                bkt.addSamples([ttZ, data, 
                        others, Zjets, Diboson, MCFakes, singletop, top])
        elif (myFitType == FitType.Exclusion) or (myFitType == FitType.Discovery):
                bkt.addSamples([data, 
                        others, Zjets, Diboson, MCFakes, ttZ, singletop, top, sigSample])
        elif options.doSignalPlots:
                print("DOING SIGNALS!")
                bkt.addSamples([data, 
                        others, Zjets, Diboson, MCFakes, ttZ, singletop, top, 
                        sigSample1, sigSample2, sigSample3, sigSample4])
        else:
                bkt.addSamples([ttZ, data, 
                        others, Zjets, Diboson, MCFakes, singletop, top])
else:   
        print("Hello ", options.createSampleHist)
        if options.createSampleHist == top.name:       bkt.addSamples([top])
        if options.createSampleHist == others.name:    bkt.addSamples([others])
        if options.createSampleHist == Zjets.name:     bkt.addSamples([Zjets])
        if options.createSampleHist == Diboson.name:   bkt.addSamples([Diboson])
        if options.createSampleHist == MCFakes.name:   bkt.addSamples([MCFakes])
        if options.createSampleHist == ttZ.name:       bkt.addSamples([ttZ])
        if options.createSampleHist == singletop.name: bkt.addSamples([singletop])


# -------------------------------------------------
# AddMeasurement
# -------------------------------------------------

meas = bkt.addMeasurement(name="NormalMeasurement", lumi=1.0, lumiErr=0.017)
meas.addPOI("mu_SIG")
meas.addParamSetting("Lumi", True, 1)

# -------------------------------------------------
# Constraining regions - statistically independent
# -------------------------------------------------

Control_region = {}
SignalRegion_region = {}
ValidationRegion_region = {}
if options.createRegionHist == '' and options.createSampleHist == '':
        for CR in Control_region_name:
                Control_region[CR] = bkt.addChannel("cuts", [CR], 1, 0.5, 1.5)
                Control_region[CR].useOverflowBin = False
                bkt.addBkgConstrainChannels([Control_region[CR]])

        if (options.doSignalPlots == False and options.doPlots == False):
                for SR in SignalRegion_region_name:
                        SignalRegion_region[SR] = bkt.addChannel("cuts",[SR],1,0.5,1.5)
                        SignalRegion_region[SR].useOverflowBin = False
                        bkt.addSignalChannels(SignalRegion_region[SR])

        if (myFitType != FitType.Exclusion) and (myFitType != FitType.Discovery) and (options.doSignalPlots == False) and (options.doPlots == False):
                for VR in ValidationRegion_region_name:
                        ValidationRegion_region[VR] = bkt.addChannel("cuts",[VR],1,0.5,1.5)
                        ValidationRegion_region[VR].useOverflowBin = False
                        bkt.addValidationChannels(ValidationRegion_region[VR])
else:
    whattocreateRegiontemp = bkt.addChannel("cuts",[options.createRegionHist],1,0.5,1.5)
    whattocreateRegiontemp.hasB = True
    whattocreateRegiontemp.hasBQCD = False
    whattocreateRegiontemp.useOverflowBin = False
    bkt.addValidationChannels([whattocreateRegiontemp])
