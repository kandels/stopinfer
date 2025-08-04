################################################################
## In principle all you have to setup is defined in this file ##
################################################################

import os
import sys
from configManager import configMgr
from ROOT import (
    kBlack,
    kWhite,
    kGray,
    kRed,
    kPink,
    kMagenta,
    kViolet,
    kBlue,
    kAzure,
    kCyan,
    kTeal,
    kGreen,
    kSpring,
    kYellow,
    kOrange,
    kDashed,
)
from configWriter import fitConfig, Measurement, Channel, Sample
from systematic import Systematic
from ROOT import gROOT, TFile, TH1F, gDirectory, TChain, SetOwnership
from time import sleep
from cutList import cutList
from optparse import OptionParser
from SetSystematic import SetSystematic
from logger import Logger

Log = Logger("HistFitter ConfigFile for ttMET 2L Stop3b Run3")


def pathResolver(Selection, sampleName):
    listFiles = []
    currenDir = ""
    # currentDir="/afs/cern.ch/user/s/skandel/stop_ANA-SUSY-2023-08/bWN1/March_DNN_ttbar_zjets/LSFHad"
    # currentDir = "/afs/cern.ch/user/s/skandel/stop_ANA-SUSY-2023-08/bWN1/MarchDNN/LSFHad/"
    # currentDir = "/afs/cern.ch/user/s/skandel/stop_ANA-SUSY-2023-08/bWN1/MayDNN/"
    currentDir = (
        "/afs/cern.ch/user/s/skandel/stop_ANA-SUSY-2023-08/bWN1/CompDNN/LSFHad/"
    )
    if sampleName == "Data":
        if Year == "20152016":
            listFiles.append(
                currentDir + "/data15_13TeV.periodAllYear.stop3b_NONE.root"
            )
            listFiles.append(
                currentDir + "/data16_13TeV.periodAllYear.stop3b_NONE.root"
            )
        elif Year == "2017":
            listFiles.append(
                currentDir + "/data17_13TeV.periodAllYear.stop3b_NONE.root"
            )
        elif Year == "2018":
            listFiles.append(
                currentDir + "/data18_13TeV.periodAllYear.stop3b_NONE.root"
            )
        elif Year == "Run2":
            listFiles.append(
                currentDir + "/data15_13TeV.periodAllYear.stop3b_NONE.root"
            )
            listFiles.append(
                currentDir + "/data16_13TeV.periodAllYear.stop3b_NONE.root"
            )
            listFiles.append(
                currentDir + "/data17_13TeV.periodAllYear.stop3b_NONE.root"
            )
            listFiles.append(
                currentDir + "/data18_13TeV.periodAllYear.stop3b_NONE.root"
            )
        elif Year == "2022":
            listFiles.append(
                currentDir + "/data22_13p6TeV.periodAllYear.stop3b_NONE.root"
            )
        elif Year == "2023":
            listFiles.append(
                currentDir + "/data23_13p6TeV.periodAllYear.stop3b_NONE.root"
            )
        elif Year == "Run3":
            listFiles.append(
                currentDir + "/data22_13p6TeV.periodAllYear.stop3b_NONE.root"
            )
            listFiles.append(
                currentDir + "/data23_13p6TeV.periodAllYear.stop3b_NONE.root"
            )
        else:
            Log.logger(
                "Wrong year given: "
                + Year
                + " -> 20152016, 2017, 2018, 2022, 2023, Run2 and Run3 can be chosen"
            )
            sys.exit(-1)

    elif sampleName == "Signals":
        if Year == "20152016":
            listFiles.append(currentDir + "/Sigstop3b.mc20a.WEIGHTS.root")
        elif Year == "2017":
            listFiles.append(currentDir + "/Sigstop3b.mc20d.WEIGHTS.root")
        elif Year == "2018":
            listFiles.append(currentDir + "/Sigstop3b.mc20e.WEIGHTS.root")
        elif Year == "Run2":
            listFiles.append(currentDir + "/Sigstop3b.mc20a.WEIGHTS.root")
            listFiles.append(currentDir + "/Sigstop3b.mc20d.WEIGHTS.root")
            listFiles.append(currentDir + "/Sigstop3b.mc20e.WEIGHTS.root")
        elif Year == "2022":
            listFiles.append(currentDir + "/Sigstop3b.mc23a.WEIGHTS.root")
        elif Year == "2023":
            listFiles.append(currentDir + "/Sigstop3b.mc23d.WEIGHTS.root")
        elif Year == "Run3":
            listFiles.append(currentDir + "/Sigstop3b.mc23a.WEIGHTS.root")
            listFiles.append(currentDir + "/Sigstop3b.mc23d.WEIGHTS.root")
        else:
            Log.logger(
                "Wrong year given: "
                + Year
                + " -> 20152016, 2017, 2018, 2022, 2023, Run2 and Run3 can be chosen"
            )
            sys.exit(-1)

    else:
        if Year == "20152016":
            listFiles.append(currentDir + "/Bkgstop3b.mc20a.WEIGHTS.root")
        elif Year == "2017":
            listFiles.append(currentDir + "/Bkgstop3b.mc20d.WEIGHTS.root")
        elif Year == "2018":
            listFiles.append(currentDir + "/Bkgstop3b.mc20e.WEIGHTS.root")
        elif Year == "Run2":
            listFiles.append(currentDir + "/Bkgstop3b.mc20a.WEIGHTS.root")
            listFiles.append(currentDir + "/Bkgstop3b.mc20d.WEIGHTS.root")
            listFiles.append(currentDir + "/Bkgstop3b.mc20e.WEIGHTS.root")
        elif Year == "2022":
            listFiles.append(currentDir + "/Bkgstop3b.mc23a.WEIGHTS.root")
        elif Year == "2023":
            listFiles.append(currentDir + "/Bkgstop3b.mc23d.WEIGHTS.root")
        elif Year == "Run3":
            listFiles.append(currentDir + "/Bkgstop3b.mc23a.WEIGHTS.root")
            listFiles.append(currentDir + "/Bkgstop3b.mc23d.WEIGHTS.root")
        else:
            Log.logger(
                "Wrong year given: "
                + Year
                + " -> 20152016, 2017, 2018, 2022, 2023, Run2 and Run3 can be chosen"
            )
            sys.exit(-1)

        print("Sample", sampleName)
        print("File considered", listFiles)

    return listFiles


def GetLumi(Year):
    lumi = 1
    if Year == "20152016":
        lumi = 3.24454 + 33.4022  # 32.9881+3.21956
    elif Year == "2017":
        lumi = 44.6306  # 44.3074
    elif Year == "2018":
        lumi = 58.7916  # 58.4501
    elif Year == "Run2":
        lumi = 58.7916 + 44.6306 + 33.4022 + 3.24454
    elif Year == "2022":
        lumi = 26.0714
    elif Year == "2023":
        lumi = 27.2399
    elif Year == "Run3":
        lumi = 27.2399 + 26.0714
    else:
        Log.logger(
            "Wrong year given: "
            + Year
            + " -> 20152016, 2017, 2018 and Run2 can be chosen"
        )
        sys.exit(-1)
    return lumi


# -------------------------------
# Parse command line
# -------------------------------
myUserArgs = configMgr.userArg.split(" ")
myInputParser = OptionParser()
myInputParser.add_option("", "--Year", dest="Year", choices=["HL", "Run2", "Run3"])
myInputParser.add_option(
    "", "--Selection", dest="Selection", choices=["Stop3b", "Stop4b"]
)
# myInputParser.add_option('', '--Channels', dest = 'Channels', default = '') #comment when using binned. uncomment for not binned
myInputParser.add_option(
    "", "--region", choices=["small", "large"], default="small"
)  # only for nbjet smallDM/largeDM model
myInputParser.add_option("", "--Syst", action="store_true", default=False)
myInputParser.add_option('--variabletoplot', dest='variabletoplot' , default='MET')
myInputParser.add_option("--doPlots"      , dest='doPlots'      , action='store_true', default=False, help="do before/after plots")

(options, args) = myInputParser.parse_args(myUserArgs)
Channels = []
# un comment for non-binned. comment for binned
# if options.Channels != "":
# Channels=options.Channels.split(",")
Year = options.Year
Selection = options.Selection
useSyst = options.Syst
# Binned nbjet
if options.region == "small":
    if Year == "Run2":
        Channels = [
            "SRRun2smallDM0b0",
            "SRRun2smallDM0b1",
            "SRRun2smallDM0b2",
            "SRRun2smallDM1b0",
            "SRRun2smallDM1b1",
        ]
    if Year == "Run3":
        Channels = [
            "SRRun3smallDM0b0",
            "SRRun3smallDM0b1",
            "SRRun3smallDM0b2",
            "SRRun3smallDM1bALL",
        ]
elif options.region == "large":
    if Year == "Run2":
        Channels = ["SRRun2largeDM0bALL", "SRRun2largeDM1b0", "SRRun2largeDM1b1"]
    if Year == "Run3":
        Channels = [
            "SRRun3largeDM0b0",
            "SRRun3largeDM0b1",
            "SRRun3largeDM1b0",
            "SRRun3largeDM1b1",
        ]

# Binned SFDF METsig > 10
# Channels = ["SRSF0b0", "SRSF0b1", "SRSF0b2", "SRSF1b0", "SRSF1b1", "SRDF0b0", "SRDF0b1", "SRDF0b2", "SRDF1b0", "SRDF1b1", "SRDF1b2"]

## First define HistFactory attributes ----

## Scaling calculated by outputLumi / inputLumi
configMgr.inputLumi = 1  # Luminosity of input TTree after weighting
lumi = GetLumi(Year)
configMgr.outputLumi = lumi
configMgr.setLumiUnits("pb-1")
configMgr.blindSR = True
configMgr.blindCR = False
configMgr.blindVR = False

## setting the parameters of the hypothesis test
configMgr.blindSR = True
configMgr.nTOYs = 20000
if HistFitterArgs.discovery_hypotest:
    configMgr.useSignalInBlindedData = True
    # configMgr.blindSR = True
configMgr.calculatorType = 2  # 2=asymptotic calculator, 0=frequentist calculator
configMgr.testStatType = (
    3  # 3=one-sided profile likelihood test statistic (LHC default)
)

sig = ""
if options.doPlots:
        configMgr.analysisName = "Distros_ttMET2L_" + Year + "_" + Selection +"_"+options.variabletoplot
else: # this could be a bug
    configMgr.analysisName = "ttMET2L_" + Year + "_" + Selection + "_" + options.region

if not HistFitterArgs.grid_points is None:
    sigSamples = HistFitterArgs.grid_points.split(",")
    if len(sigSamples) == 1:
        configMgr.analysisName += "_" + sigSamples[0]
        sig = sigSamples[0]
        log.info("Analyzing signal " + sig + " with selection" + Selection)
    else:
        log.error("please run one signal per job and parallelize jobs")
        sys.exit(-1)
else:
    print(configMgr.analysisName)
    configMgr.analysisName += "_OnlyBkg"

for channel in Channels:
    configMgr.analysisName += "_" + channel

if HistFitterArgs.hypotest:
    configMgr.analysisName += "_exclusion"
if HistFitterArgs.discovery_hypotest:
    configMgr.analysisName += "_discovery"

configMgr.outputFileName = "results/" + configMgr.analysisName + ".root"
print('Analysis Name ', configMgr.analysisName)

## Set the cache file
print("Cache file: ", configMgr.histBackupCacheFile)
configMgr.useCacheToTreeFallback = True
configMgr.useHistBackupCacheFile = False
configMgr.histCacheFile = "data/CacheFile_" + configMgr.analysisName + ".root"
configMgr.histBackupCacheFile = (
    os.environ["ttMET2LRun3Analysis"]
    + "/InputHF/"
    + Selection
    + "/CacheFile_ttMET2L_"
    + Year
    + "_"
    + Selection
    + "_"
    + options.region
    + "_OnlyBkg.root"
)

## Suffix of nominal tree
configMgr.nomName = "_WEIGHTS"

##weight
weights = [
    "WeightEvents",
    "xsec",
    "WeightEventsPU",
    "WeightLumi",
    "WeightEventselSF",
    "WeightEventsmuSF",
    "WeightEventsJVT",
    "WeightEventsbTag",
    "WeightEventsSF_global",
]
configMgr.weights = weights

# workspace configuration
Type = ""
if sig:
    Type = sig
else:
    Type = "FitConfig"
Type += "_" + Selection
for channel in Channels:
    Type += "_" + channel

myTopLvl = configMgr.addFitConfig(Type)
myTopLvl.statErrThreshold = 0.00001

# Parameters of the Measurement
measName = "BasicMeasurement"
measLumi = 1.0
measLumiError = 0.017
if Year == "HL":
    measLumiError *= 0.6

## Parameters of Channels
meas = myTopLvl.addMeasurement(
    measName, measLumi, measLumiError
)  # Set lumi and its error
# meas.addParamSetting("Lumi",True,1) #Neclect the lumi NP
meas.addPOI("mu_SIG")

##Legend
myTopLvl.totalPdfColor = kBlack
myTopLvl.getDataColor = kBlack
myTopLvl.errorFillStyle = 3325
myTopLvl.errorLineStyle = kDashed
myTopLvl.errorLineColor = kBlue

##addSamples with prompt leptons
samplesMaps = {}
bkgSamples = []
samplesMaps = {
    "ttbar": ["ttbar", kYellow],
    "Wt_dyn_DR": ["Wt_dyn_DR", kRed],
    "Zjets_Sh": ["Zjets_Sh", kSpring],
    "VV": ["VV", kViolet + 1],
    "ttZ": ["ttZ", kBlue + 1],
    "Others": ["other", kTeal + 4],
    "data": ["data_NONE", kBlack],
}
bkgSamples = ["ttbar", "VV", "ttZ", "Wt_dyn_DR", "Zjets_Sh", "Others"]

##LoadMapRegions
cList = cutList(Selection)
configMgr.cutsDict = cList.cutsDict
###IsoSignal, Prompt, Fake Weight
IsoSig = cList.IsoSigWeight
PromptLep = cList.PromptOriginWeight
FakeLep = cList.FakeOriginWeight
BadEvents = cList.BadEvents


###Initialize Syst Class
syst = SetSystematic()
syst.FakesSamples = ["Fakes"]
syst.systNONE = configMgr.nomName
syst.cutsList = cList
if (
    not configMgr.histBackupCacheFile == ""
    and not configMgr.readFromTree
    and configMgr.useHistBackupCacheFile    
):
    syst.SetCacheFile(configMgr.histBackupCacheFile)
# SR
SRs = []
for channel in Channels:
    SR = myTopLvl.addChannel("cuts", [channel], 1, 0.5, 1.5)
    SRs.append(SR)
myTopLvl.addSignalChannels(SRs)

###
if options.doPlots:
    d = {}
    rangeplot = {}
    rangeplot["isSF"]          = [2 , -0.5, 1.5]
    rangeplot["nbjet"]         = [6 , -0.5, 5.5]
    rangeplot["njet"]          = [6 , -0.5, 5.5]
    rangeplot["lep1pT"]        = [50, 0.  , 500.]
    rangeplot["lep2pT"]        = [10, 0.  , 500.]
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
    rangeplot["MET"]           = [10, 0 , 500.]
    rangeplot["METsig"]        = [20, 0 , 20.]
    rangeplot["RPT"]           = [10, 0   , 1]
    rangeplot["gamInvRp1"]     = [10, 0   , 1]
    rangeplot["MDR"]           = [40, 0   , 400]
    rangeplot["DPB_vSS"]       = [32, 0.  , 3.2]
    rangeplot["cosTheta_b"]    = [10, 0   , 1]
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
    rangeplot["DNN_smallDM_bjets1_ttbar"] = [25, 0, 1]
    rangeplot["DNN_largeDM_bjets1_ttbar"] = [25, 0, 1]
    rangeplot["DNN_smallDM_bjets0_ttbar"] = [25, 0, 1] # just for validtion
    rangeplot["DNN_largeDM_bjets0_ttbar"] = [25, 0, 1] # just for validation

    # add VR later
    CRTOP =  myTopLvl.addChannel(options.variabletoplot,[f"CRTOP{options.region}1b{Year}"], rangeplot[options.variabletoplot][0], rangeplot[options.variabletoplot][1], rangeplot[options.variabletoplot][2])
    CRVV =  myTopLvl.addChannel(options.variabletoplot,[f"CRVV{options.region}0b{Year}"], rangeplot[options.variabletoplot][0], rangeplot[options.variabletoplot][1], rangeplot[options.variabletoplot][2])
    #VRTOP =  myTopLvl.addChannel(options.variabletoplot,[f"VRTOP{options.region}1b{Year}"], rangeplot[options.variabletoplot][0], rangeplot[options.variabletoplot][1], rangeplot[options.variabletoplot][2])
    #VRVV =  myTopLvl.addChannel(options.variabletoplot,[f"VRVV{options.region}0b{Year}"], rangeplot[options.variabletoplot][0], rangeplot[options.variabletoplot][1], rangeplot[options.variabletoplot][2])
    CRTOP.useOverflowBin = True
    CRVV.useOverflowBin = True
    #VRTOP.useOverflowBin = True
    #VRVV.useOverflowBin = True
    myTopLvl.addValidationChannels([CRTOP, CRVV])
###

# CRs
CRs = []
# Only create CRs that are actually used for normalization
cr_channels = [f"CRTOP{options.region}1b{Year}", f"CRVV{options.region}0b{Year}"]
for channel in cr_channels:
    CR = myTopLvl.addChannel("cuts", [channel], 1, 0.5, 1.5)
    CRs.append(CR)
myTopLvl.addBkgConstrainChannels(CRs)

###Samples
for sample in bkgSamples:
    log.info("Taking " + sample + " sample")
    s = Sample(sample, samplesMaps[sample][1])
    s.prefixTreeName = samplesMaps[sample][0]
    s.addInputs(pathResolver(Selection, sample))
    s.setStatConfig(True)
    s.addSampleSpecificWeight(cList.isPrompt)
    if sample == "ttbar":
        s.addSampleSpecificWeight("(nlep==2)")
        norm_regions = [(f"CRTOP{options.region}1b{Year}", "cuts")]
        s.setNormRegions(norm_regions)
        s.setNormFactor("mu_top", 1.0, 0.0, 5.0)
    elif sample == "VV":
        norm_regions = [(f"CRVV{options.region}0b{Year}", "cuts")]
        s.setNormRegions(norm_regions)
        s.setNormFactor("mu_VV", 1.0, 0.0, 5.0)
    elif sample == "MCFakes":
        s.addSampleSpecificWeight(cList.isNOPrompt)
        s.setNormByTheory()
    else:
        s.setNormByTheory()

    # addCuts = []
    #addCuts.append(PromptLep)
    # should I use it for all samples? 4b and 2b do this for all. 2b - except MCFakes
    #addCuts.append(IsoSig)
    # addCuts.append(BadEvents)
    # s.additionalCuts = "( " + " && ".join(addCuts) + " )"
    # normRegions = [(sr.regionString, sr.variableName) for sr in SRs]
    # print(normRegions)
    # s.setNormRegions(normRegions)
    # myTopLvl.addSamples(s) #it was here before messing up with 20% syst
    if useSyst:
        flatSyst = Systematic(
            name=f"BkgNorm_{sample}",
            nominal=1,
            high=1.2,
            low=0.8,
            type="user",
            method="userOverallSys",
        )
        s.addSystematic(flatSyst)
        log.info(f"Added {flatSyst.name} systematic to {sample}")
    myTopLvl.addSamples(s)


# removed fakes

####Data
dataSample = Sample("data", samplesMaps["data"][1])
dataSample.prefixTreeName = samplesMaps["data"][0]
dataSample.addInputs(pathResolver(Selection, "Data"))
#dataSample.additionalCuts = IsoSig
dataSample.setData()
dataSample.setStatConfig(True)
myTopLvl.addSamples(dataSample)


if HistFitterArgs.grid_points:
    if len(sigSamples) == 1:
        sigName = "Signal_" + sig.split("_")[-2] + "_" + sig.split("_")[-1]
        sigSample = Sample(sigName, kBlue)
        sigSample.setPrefixTreeName(sig)
        sigSample.addInputs(pathResolver(Selection, "Signals"))
        sigSample.setNormFactor("mu_SIG", 1.0, 0.0, 15.0)
        #sigSample.addSampleSpecificWeight(PromptLep)
        sigSample.addSampleSpecificWeight(cList.isPrompt)
        #sigSample.addSampleSpecificWeight(IsoSig)
        sigSample.setStatConfig(True)
        myTopLvl.addSamples(sigSample)
        myTopLvl.setSignalSample(sigName)
    # commented these lines to use --Syst only on BKG for 20% syst.
    # if len(SRs)>0:
    # for sr in SRs:   #Not th syst needed for signal recast
    # syst.AddThSyst(sigName,sr)
