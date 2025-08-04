import ROOT
from optparse import OptionParser
import contourPlotter

#PARSING USER INPUT

myContourParser = OptionParser()
myContourParser.add_option("", "--Year", action="store", type="string", dest="Year")
myContourParser.add_option("", "--YLabel", action="store", type="string",dest = "YLabel")
myContourParser.add_option("", "--xMin", action="store", type = "int", dest = "xMin", default = 258)
myContourParser.add_option("", "--xMax", action="store", type = "int",dest = "xMax", default = 800)
myContourParser.add_option("", "--yMin", action="store", type = "int",dest = "yMin", default = 0)
myContourParser.add_option("", "--yMax", action="store", type = "int",dest = "yMax", default = 750)
myContourParser.add_option("", "--drawTheorySystematics", action = "store", dest = "drawTheorySystematics", default = False)
myContourParser.add_option("","--TypeCLs", action = "store", dest = "TypeCLs", default = "Obs")
myContourParser.add_option("", "--ID", type="string", dest="ID", default="DNN")
(options,args) = myContourParser.parse_args()

Year = options.Year
YLabel = options.YLabel
drawTheorySysts = options.drawTheorySystematics
axis_boundary = {
    'mchi': [258, 0, 800, 750],
    'DM': [258, 50, 800, 300]
}
xMin = options.xMin
xMax = options.xMax
yMin = options.yMin
yMax = options.yMax
TypeCLs = options.TypeCLs
ID = options.ID

if YLabel == "mchi":
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/RPT_0_70_cut/mstop_mchi.root"
    f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/RPT_0_70_cut_binned/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/RPT_binned_nbjet/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/SFDF_unbinned/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/SFDF_metsig_unbinned/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/SFDF_binned/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/sfdf_binned_20_sys/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/rpt_metsig/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/StrongJune25/roots/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/StrongJune25/json/Run3/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/StrongJune25/json/Run2cache/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/bin_run2run3/run2/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/withLeptrigger/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/CR_Jul23/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/CR_Jul23v1/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/CR_Jul23v2/mstop_mchi.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/CR_Jul23v2/R3/mstop_mchi.root"
    f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/CR_Aug1/Run3/mstop_mchi.root"
    f_run2 = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/Run2/mstop_mchi.root"
elif YLabel == "DM":
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/RPT_0_70_cut/mstop_dm.root"
    f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/RPT_0_70_cut_binned/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/RPT_binned_nbjet/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/SFDF_unbinned/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/SFDF_metsig_unbinned/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/SFDF_binned/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/sfdf_binned_20_sys/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/rpt_metsig/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/StrongJune25/roots/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/StrongJune25/json/Run3/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/StrongJune25/json/Run2cache/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/bin_run2run3/run3/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/withLeptrigger/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/CR_Jul23/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/CR_Jul23v1/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/CR_Jul23v2/mstop_dm.root"
    #f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/CR_Jul23v2/R3/mstop_dm.root"
    f_name = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/CR_Aug1/Run3/mstop_dm.root"
    f_run2 = "/afs/cern.ch/user/s/skandel/skandel/Run3ttMET/HMBS-2024-10_interpretation/jsons/Run2/mstop_dm.root"
else:
    print("Incorrect YLabel", YLabel)
    exit()

#drawTheorySysts = False


plot = contourPlotter.contourPlotter("contourPlotterExample",800,600)
plot.plotName = f"mstop_{YLabel}_{Year}_{ID}"
plot.figLabel = "ATLAS Internal"

#plot.processLabel = "#tilde{t}_{1}#tilde{t}_{1}#rightarrow t#chi_{1}^{0}, #tilde{t}_{1}#tilde{t}_{1} #rightarrow bW#chi_{1}^{0}"
plot.processLabel = "#tilde{t}_{1}#tilde{t}_{1} #rightarrow bW#chi_{1}^{0}"

if Year == "Run2":
    plot.lumiLabel = "#scale[0.8]{#sqrt{s}=13 TeV, 139 fb^{-1}, Limits at 95% CL}"
elif Year == "Run3":
    plot.lumiLabel = "#scale[0.8]{#sqrt{s}=13.6 TeV, 53 fb^{-1}, Limits at 95% CL}"
else:
    plot.lumiLabel = "#scale[0.5]{#sqrt{s}=14 TeV, 3000 fb^{-1}, Limits at 95% CL}"

fRun2 = ROOT.TFile(f_run2)
fRun2_rel22 = ROOT.TFile(f_name)

## Axes
plot.drawAxes(axis_boundary[YLabel])
print("Axis range:", xMin, xMax, yMin, yMax)

plot.drawExpected(fRun2_rel22.Get("Exp_0"), title="DNN")
plot.drawExpected(fRun2.Get("Exp_0"), title="Run2 Expected", legendOrder=0, color=ROOT.kRed)
#plot.drawExpected(fRun2.Get("Exp_0"), title="#scale[0.5]{Run2 Observed Limit (3-body only)}", legendOrder=0, color=ROOT.kRed)
plot.drawObserved(fRun2.Get("Obs_0"), title="#scale[0.5]{Run2 Observed Limit (3-body only)}", legendOrder=0, color=ROOT.kRed)
plot.drawOneSigmaBand(  fRun2_rel22.Get("Band_1s_0"))
plot.drawTextFromTGraph2D( fRun2_rel22.Get("CLsexp_gr")  , angle=30 , title = "Grey Numbers Represent Expected CLs Value")

plot.setXAxisLabel( "m(#tilde{t}_{1}) [GeV]" )
if YLabel == "DM":
    plot.setYAxisLabel("#Delta m(#tilde{t}_{1}, #tilde{#chi_{1}^{0}}) [GeV]")
elif YLabel == "mchi":
    plot.setYAxisLabel("m(#tilde{#chi_{1}^{0}}) [GeV]")

forb_coord = {
    'mchi': ([258, 83, 800, 625], [258, 175, 800, 717], [450, 250], [450,380], 30),
    'DM': ([258,172, 800, 172], [258,83, 800, 83], [500, 75], [500, 180], 0)
}

plot.createLegend(shape=(0.22,0.58,0.55,0.77) ).Draw()

plot.drawLine(coordinates = forb_coord[YLabel][0], label="#Delta m = 85 GeV", labelLocation=forb_coord[YLabel][2], style=7, color=ROOT.kGray+2, angle=forb_coord[YLabel][4])
plot.drawLine(coordinates = forb_coord[YLabel][1], label="#Delta m = 172 GeV",  labelLocation=forb_coord[YLabel][3], style=7, color=ROOT.kGray+2, angle=forb_coord[YLabel][4])

if drawTheorySysts:
	plot.drawTheoryUncertaintyCurve( f.Get("Obs_0_Up") )
	plot.drawTheoryUncertaintyCurve( f.Get("Obs_0_Down") )
	# coordinate in NDC
	plot.drawTheoryLegendLines( xyCoord=(0.234,0.6625), length=0.057 )

plot.decorateCanvas( )
plot.writePlot()
