import os
import sys
import numpy as _np
import math
try:
    import rpnpy.vgd.all as _vgd
except ImportError:
    sys.exit("\n\nPlease load a rpnpy ssm package for"
             + " python3, look for rpnpy on the wiki.\n\n")
import rpnpy.librmn.all as rmn


class MyError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class PlotParameters:
    """Set plot parameters"""
    def __init__(self, ymin, ymax):
        self.ymin = ymin
        self.ymax = ymax
        _decade = _np.array([100, 70, 50, 35, 25, 15])
        self.y_ticks_v = _np.array(
            [1013.25, 950, 900, 850, 800, 700, 600, 500, 420, 350, 300, 250, 
             200, 160, 130])
        self.y_ticks_v = _np.append(self.y_ticks_v, _decade)
        self.y_ticks_v = _np.append(self.y_ticks_v, _decade*.1)
        self.y_ticks_v = _np.append(self.y_ticks_v, _decade*.01)
        self.y_ticks_v = _np.append(self.y_ticks_v, _decade*.001)
        self.y_ticks_v = _np.extract(self.y_ticks_v+0.001 >= 
                                    math.exp(self.ymin)/100.,
                                    self.y_ticks_v)
        self.y_ticks_v = _np.extract(self.y_ticks_v-0.001 <= 
                                    math.exp(self.ymax)/100.,
                                    self.y_ticks_v)
        if ymin < self.y_ticks_v[-1]:
             self.y_ticks_v = _np.append(self.y_ticks_v, round(_np.exp(ymin))/100.)
        self.y_ticks_name = self.y_ticks_v
        self.y_ticks_v = _np.log(self.y_ticks_v*100.)

class Topo:
    """Generate topo field for vgrid"""
    def __init__(self, pbot_hill, ptop_hill, ni, ss_amp = 0,
                 ss_cycles = 11.):
        self.xx = _np.float32([i*2*math.pi/ni for i in _np.linspace(0, ni, ni)])
        self.p0 = _np.linspace(0, ni, ni)
        self.p0 = pbot_hill-(.5*(1.-_np.cos(self.xx))
                                * (pbot_hill-ptop_hill))
        self.p0ls = _np.copy(self.p0)
        self.mels = _vgd.vgd_stda76_hgts_from_pres_list(self.p0ls)
        _modulation = _np.copy(self.p0)
        if self.p0[0] - ptop_hill != 0:
            _modulation = 1. - (self.p0 - ptop_hill)/(self.p0[0] - ptop_hill)
        else:
            _modulation = 0.
        _tempo = _np.copy(self.p0 + _modulation * .5 * ss_amp 
                          * (1. - _np.cos(self.xx*ss_cycles)))
        self.p0 = _np.copy(_tempo)
        self.me = _vgd.vgd_stda76_hgts_from_pres_list(self.p0)

class Vertical:
    
    """Vertical grid information class"""

    IP1_HYB1_P = 93423264
    IP1_HYB1_H = 93423364

    def __init__(self, file):
        self.file = file
        self.ni = 0
        self.nk = 0
        self.nkt = 0
        self.nkw = 0
        self.vgd = None
        self.ip1m = None
        self.levels = None
        self.levelst = None
        self.levelsw = None
        self.levels_delta = None
        self.val = None
        self.valt = None
        self.valw = None
        self.kind = None
        self.kindt = None
        self.kindw = None        
        self.ip1t = None
        self.sfc = None
        self.sfc_ls = None
        self.in_log = None
        self.to_pres = None
        try:
            _fileid = rmn.fstopenall(file, rmn.FST_RO)
        except:
            raise MyError("There was a problem opening file " + self.file
                          + "\n")
        try:
            self.vgd = _vgd.vgd_read(_fileid)
        except:
            raise MyError("There was a problem reading the VGridDescriptor "
                          + "for file " + self.file + "\n")
        finally:
            rmn.fstcloseall(_fileid)
        
    def VGD_get_levels(self, topo):
        """Compute levels"""
        # All scale sfc data
        try: 
            rfld = _vgd.vgd_get(self.vgd, 'RFLD')
        except:
            raise
        if rfld == "P0":
            self.sfc = topo.p0
            self.in_log = 1
            self.to_pres = 0
        elif rfld == "ME":
            self.sfc = topo.me
            self.in_log = 0
            self.to_pres = 1
        else:
            raise ValueError("Execpting P0 or ME, got " + rfld)
        # Large scale sfc data
        try: 
            rfls = _vgd.vgd_get(self.vgd, 'RFLS')
        except:
            rfls = None
        else:
            if rfls == "P0LS":
                self.sfc_ls = topo.p0ls
            elif rfls == "MELS":
                self.sfc_ls = topo.mels
            else:
                raise ValueError("Execpting P0LS or MELS, got " + rfls)
        # Problem size
        self.nk = _vgd.vgd_get(self.vgd, 'NL_M')
        self.ni = self.sfc.size
        # Momentum ip1
        try:
            self.ip1m = _vgd.vgd_get(self.vgd, 'VIPM')
        except:
            raise MyError("There was a problem getting vgd parameter VIPM for "
                          + "file " + self.file + "\n")
        # Compute momentum levels
        try:
            self.levels = _vgd.vgd_levels(self.vgd, ip1list=self.ip1m,
                                         rfld=self.sfc, rfls=self.sfc_ls,
                                         in_log=self.in_log)
        except _vgd.VGDError:
            raise MyError("There was a problem computing Momentum "
                          + "(full) levels for file "
                          + self.file + "\n")
        if self.to_pres:
            try:
                _tempo = _vgd.vgd_stda76_pres_from_hgts_list(self.levels)
            except:
                print("Error computing pressure from height list for momentum "
                      + "levels")
                raise
            self.levels = _np.copy(_np.log(_tempo))
        self.val = _np.zeros(self.nk, dtype=_np.float)
        self.kind = _np.zeros(self.nk, dtype=_np.int)
        for k in range(0, self.nk):
            (self.val[k], self.kind[k]) = rmn.convertIp(rmn.CONVIP_DECODE,
                                                        self.ip1m[k])

        # Compute thermo levels
        self.nkt = _vgd.vgd_get(self.vgd, 'NL_t')
        try:
            self.ip1t = _vgd.vgd_get(self.vgd, 'VIPT')
        except:
            raise MyError("There was a problem getting vgd parameter VIPT for "
                          + "file " + self.file + "\n")
        if _np.array_equal(self.ip1m, self.ip1t):
            # There are no thermo level, python rpn only gives momentum ip1 to 
            # avoid error we set self.nkt to zero so it can be used for testing
            # thermo presence
            self.nkt = 0
        else:
            try:
                self.levelst = _vgd.vgd_levels(self.vgd, ip1list=self.ip1t,
                                              rfld=self.sfc, rfls=self.sfc_ls,
                                              in_log=self.in_log)
            except _vgd.VGDError:
                raise MyError("There was a problem computing Thermo levels for"
                              + "file " + self.file + "\n")
            if self.to_pres:
                try:
                    _tempo = _vgd.vgd_stda76_pres_from_hgts_list(self.levelst)
                except:
                    print("Error computing pressure from height list for"
                          + " thermo levels")
                    raise
                self.levelst = _np.copy(_np.log(_tempo))
            self.valt = _np.zeros(self.nkt, dtype=_np.float)
            self.kindt = _np.zeros(self.nkt, dtype=_np.int)
            for k in range(0, self.nkt):
                (self.valt[k], self.kindt[k]) = rmn.convertIp(rmn.CONVIP_DECODE,
                                                              self.ip1t[k])
        # Compute vertical volocyty levels
        self.nkw = _vgd.vgd_get(self.vgd, 'NL_W')
        try:
            self.ip1w = _vgd.vgd_get(self.vgd, 'VIPW')
        except:
            raise MyError("There was a problem getting vgd parameter VIPW for "
                          + "file " + self.file + "\n")
        if _np.array_equal(self.ip1m, self.ip1w):
            # There are no vertical velocity  level, python rpn only gives
            # momentum ip1 to avoid error we set self.nkt to zero so it can
            # be used for testing thermo presence
            self.nkw = 0
        else:
            try:
                self.levelsw = _vgd.vgd_levels(self.vgd, ip1list=self.ip1w,
                                              rfld=self.sfc, rfls=self.sfc_ls,
                                              in_log=self.in_log)
            except _vgd.VGDError:
                raise MyError("There was a problem computing vertica velocity"
                              + " levels for file " + self.file + "\n")
            if self.to_pres:
                try:
                    _tempo = _vgd.vgd_stda76_pres_from_hgts_list(self.levelsw)
                except:
                    print("Error computing pressure from height list for"
                          + " vertical velocity")
                    raise
                self.levelsw = _np.copy(_np.log(_tempo))
            self.valw = _np.zeros(self.nkw, dtype=_np.float)
            self.kindw = _np.zeros(self.nkw, dtype=_np.int)
            for k in range(0, self.nkw):
                (self.valw[k], self.kindw[k]) = rmn.convertIp(rmn.CONVIP_DECODE,
                                                              self.ip1w[k])

    def VGD_get_levels_delta(self):
        """Compute levels resolution (delta)"""
        if not self.nk:
            raise MyError("ERROR with vgd_levels_delta, levels are not "
                          + "computed for file " + self.file + "\n")
        self.levels_delta = _np.zeros((self.ni, self.nk-1), dtype=_np.float)
        self.levels_delta = (self.levels[:, 1:self.nk] 
                             - self.levels[:, 0:(self.nk-1)])

    def VGD_get_vcode(self):
        """Return vertical coordonate Vcode"""
        return 1000 * _vgd.vgd_get(self.vgd, 'KIND') + _vgd.vgd_get(self.vgd,
                                                                 'VERSION')

    def VGD_get_ptop(self):
        """Return vertical coordonate ptop as a list, may be empty"""
        ptop = []
        try:
            value = _vgd.vgd_get(self.vgd, 'PTOP', quiet=1)
        except:
            value = _vgd.VGD_MISSING
        if value != _vgd.VGD_MISSING:
            ptop.append(value)
        return ptop

    def VGD_get_pref(self):
        """Return vertical coordonate pref as a list, may be empty"""
        pref = []
        try:
            value = _vgd.vgd_get(self.vgd, "PREF", quiet=1)
        except:
            value = _vgd.VGD_MISSING
        if value != _vgd.VGD_MISSING:
            pref.append(value)
        return pref

    def VGD_get_rcoefs(self):
        """Return vertical coordonate rcoefs as a list, may be empty"""
        rcoefs = []
        value = _vgd.vgd_get(self.vgd, "RC_1")
        if value != _vgd.VGD_MISSING:
            rcoefs.append(value)
        value = _vgd.vgd_get(self.vgd, "RC_2")
        if value != _vgd.VGD_MISSING:
            rcoefs.append(value)
        try:
            value = _vgd.vgd_get(self.vgd, "RC_3")
        except:
            pass
        else:
            if value != _vgd.VGD_MISSING:
                rcoefs.append(value)
        try:
            value = _vgd.vgd_get(self.vgd, "RC_4")
        except:
            pass
        else:
            if value != _vgd.VGD_MISSING:
                rcoefs.append(value)
        return rcoefs


def plot_it(args, i, plt, gs, gs_legend, vert, topo, ppar, title):

    fontsize=12*args.scale_fonts

    if title == "ETIKET":
        try:
            title = _vgd.vgd_get(vert.vgd, 'ETIKET')
        except:
            print("Problem retreiving etiket from vgrid")
            raise
    ax = plt.subplot(gs)
    ax.set_title(title, size=12*args.scale_fonts)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    xmin = _np.min(topo.xx)
    xmax = _np.max(topo.xx)
    plt.xlim((xmin, xmax))
    plt.ylim((ppar.ymax, ppar.ymin))
    plt.tick_params(axis='x', which='both', bottom='off', top='off',
                    labelbottom='off')
    if i == 0:
        plt.ylabel('Pressure [hPa]', fontsize=fontsize)
    else:
        plt.tick_params(axis='y', labelleft='off')

    # Plot momentum levels
    plt.yticks(ppar.y_ticks_v, ppar.y_ticks_name, fontsize=fontsize)
    ax.set_color_cycle(['black'])
    for k in range(vert.nk):
        # Do not trace hyb = 1.0
        if vert.ip1m[k] != vert.IP1_HYB1_P and vert.ip1m[k] != vert.IP1_HYB1_H:
            if vert.kind[k] == 4:
                if not args.hide_diag_levels:
                    ax.set_color_cycle(['orange'])      
                    ax.plot(topo.xx, vert.levels[:, k])
            else:
                ax.plot(topo.xx, vert.levels[:, k])
        # Trace hyb = 1.0 if 5001
        if vert.ip1m[k] == vert.IP1_HYB1_P and vert.VGD_get_vcode() == 5001:
            ax.set_color_cycle(['black'])
            ax.plot(topo.xx, vert.levels[:, k])

    # Plot thermo levels
    if args.plot_thermo and vert.nkt:
        ax.set_color_cycle(['red'])
        for k in range(vert.nkt):
            if (vert.ip1t[k] != vert.IP1_HYB1_P
                and vert.ip1t[k] != vert.IP1_HYB1_H):
                if vert.kindt[k] == 4:
                    if not args.hide_diag_levels:
                        ax.set_color_cycle(['orange'])
                        ax.plot(topo.xx, vert.levelst[:, k], '--')
                else:
                    ax.plot(topo.xx, vert.levelst[:, k], '--')
    # Plot vertical velocity levels
    if args.plot_vertical_velocity and vert.nkw:
        ax.set_color_cycle(['blue'])
        for k in range(vert.nkw):
            if (vert.ip1w[k] != vert.IP1_HYB1_P
                and vert.ip1w[k] != vert.IP1_HYB1_H):
                if vert.kindw[k] == 4:
                    if not args.hide_diag_levels:
                        ax.set_color_cycle(['orange'])
                        ax.plot(topo.xx, vert.levelsw[:, k], ':')
                else:
                    ax.plot(topo.xx, vert.levelsw[:, k], ':')
            # Trace hyb = 1.0 if 21002
            if vert.ip1m[k] == vert.IP1_HYB1_H and vert.VGD_get_vcode() == 21002:
                ax.set_color_cycle(['black'])
                ax.plot(topo.xx, vert.levels[:, k],':')
    # Plot topo
    ax.set_color_cycle(['green'])
    ax.plot(topo.xx, _np.log(topo.p0), '--', linewidth=2)

    # Legend
    ax_legend = plt.subplot(gs_legend)
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    plt.axis('off')
    plt.xlim((0., 1.))
    plt.ylim((1., 0.))

    line_spacing = args.legend_text_spacing
    i = 1
    # Vcode
    if args.vcode:
        check_legend(args, i, line_spacing)                
        ax_legend.text(.5, i*line_spacing, "Vcode "
                       + str(vert.VGD_get_vcode()),
                       horizontalalignment='center', 
                       fontsize=fontsize)
        i = i + 1
    # Momentum levels
    check_legend(args, i, line_spacing)
    ax_legend.set_color_cycle(['black'])
    ax_legend.plot((.2,.8), ((i+.05)*line_spacing, (i+.05)*line_spacing))
    ax_legend.text(.5, i*line_spacing, str(vert.nk) +
                   " momentum levels", horizontalalignment='center',
                   fontsize=fontsize)
    i = i + 1
    # Topo
    check_legend(args, i, line_spacing)
    ax_legend.set_color_cycle(['green'])
    ax_legend.plot((.2,.8), ((i+.05)*line_spacing, (i+.05)*line_spacing), '--',
                   linewidth=2)
    ax_legend.text(.5, i*line_spacing,"topo", horizontalalignment='center',
                   fontsize=fontsize)
    i = i + 1
    # Thermo levels    
    if args.plot_thermo and vert.nkt:
        check_legend(args, i, line_spacing)
        ax_legend.set_color_cycle(['red'])
        ax_legend.plot((.2,.8), ((i+.05)*line_spacing, (i+.05)*line_spacing),
                       '--')
        ax_legend.text(.5, i*line_spacing,
                       "Thermo levels",
                       horizontalalignment='center',
                       fontsize=fontsize)
        i = i + 1
    # Vert velocity levels
    if args.plot_vertical_velocity and vert.nkw:
        ax_legend.set_color_cycle(['blue'])
        ax_legend.plot((.2,.8), ((i+.05)*line_spacing, (i+.05)*line_spacing),
                       ':')
        ax_legend.text(.5, i*line_spacing,
                       "Vert velocity levels",
                       horizontalalignment='center',
                       fontsize=fontsize)
        i = i + 1
    # Diag m level
    if not args.hide_diag_levels:
        for k in range(vert.nk):
            if vert.kind[k] == 4:
                check_legend(args, i, line_spacing)
                ax_legend.set_color_cycle(['orange'])
                ax_legend.plot((.2,.8), ((i+.05)*line_spacing,
                                         (i+.05)*line_spacing))
                ax_legend.text(.5, i*line_spacing,
                               "Diag m level at " + str(vert.val[k]) + "m",
                               horizontalalignment='center',
                               fontsize=fontsize)
                i = i + 1
    # Diag t level
    if not args.hide_diag_levels:
        if args.plot_thermo and vert.nkt:
            for k in range(vert.nkt):
                if vert.kindt[k] == 4:
                    check_legend(args, i, line_spacing)
                    ax_legend.set_color_cycle(['orange'])
                    ax_legend.plot((.2,.8), ((i+.05)*line_spacing,
                                             (i+.05)*line_spacing), '--')
                    ax_legend.text(.5, i*line_spacing,
                                   "Diag t level at " + str(vert.valt[k]) + "m",
                                   horizontalalignment='center',
                                   fontsize=fontsize)
                    i = i + 1
    # Diag w level
    if not args.hide_diag_levels:
        if args.plot_vertical_velocity and vert.nkw:
            for k in range(vert.nkw):
                if vert.kindw[k] == 4:
                    check_legend(args, i, line_spacing)
                    ax_legend.set_color_cycle(['orange'])
                    ax_legend.plot((.2,.8), ((i+.05)*line_spacing,
                                             (i+.05)*line_spacing), ':')
                    ax_legend.text(.5, i*line_spacing,
                                   "Diag w level at " + str(vert.valw[k]) + "m",
                                   horizontalalignment='center',
                                   fontsize=fontsize)
                    i = i + 1
    # Ptop
    ptop = vert.VGD_get_ptop()
    if ptop:
        check_legend(args, i, line_spacing)
        ax_legend.text(.5, i*line_spacing,
                       "ptop " + str(ptop),
                       horizontalalignment='center',
                       fontsize=fontsize)
        i = i + 1
    # Pref
    pref = vert.VGD_get_pref()
    if pref:
        check_legend(args, i, line_spacing)
        ax_legend.text(.5, i*line_spacing,
                       "pref " + str(pref),
                       horizontalalignment='center',
                       fontsize=fontsize)
        i = i + 1
    # rcoefs
    rcoefs = vert.VGD_get_rcoefs()
    if rcoefs:
        check_legend(args, i, line_spacing)
        ax_legend.text(.5, i*line_spacing,
                       "Rcoef(s) " + str(rcoefs),
                       horizontalalignment='center',
                       fontsize=fontsize)
        i = i + 1

def plot_it_delta(args, plt, gs, gs_legend, vert, topo, ppar, xmin, xmax):

    fontsize=12*args.scale_fonts

    ax = plt.subplot(gs, title="Levels resolution")
    plt.xlim((xmin, xmax))
    plt.ylim((ppar.ymax, ppar.ymin))
    plt.tick_params(axis='x', which='both', bottom='on', top='off',
                    labelbottom='on')
    plt.tick_params(axis='y', labelleft='off')
    plt.xlabel('Delta in ln pressure')

    # Plot delta
    plt.yticks(ppar.y_ticks_v, ppar.y_ticks_name)
    #delta = _np.amax(vert.levels_delta, axis=0)
    ax.set_color_cycle(['blue'])
    if args.plot_delta_symb:
        ax.plot(vert.levels_delta[0,:], .5*(vert.levels[0, 0:(vert.nk-1)]
                                        + vert.levels[0, 1:vert.nk]),marker='o')
    else:
        ax.plot(vert.levels_delta[0,:], .5*(vert.levels[0, 0:(vert.nk-1)]
                                        + vert.levels[0, 1:vert.nk]))
        
    # Find top of hill
    imin = _np.argmin(topo.p0)
    ax.set_color_cycle(['red'])
    if args.plot_delta_symb:
        ax.plot(vert.levels_delta[imin,:], .5*(vert.levels[imin, 0:(vert.nk-1)]
                                    + vert.levels[imin, 1:vert.nk]),marker='s')
    else:
        ax.plot(vert.levels_delta[imin,:], .5*(vert.levels[imin, 0:(vert.nk-1)]
                                    + vert.levels[imin, 1:vert.nk]))        

    # Legend
    ax_legend = plt.subplot(gs_legend)
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    plt.axis('off')
    plt.xlim((0., 1.))
    plt.ylim((1., 0.))

    line_spacing = args.legend_text_spacing
    i = 4
    ax_legend.text(.5, i*line_spacing, "Blue: delta below hill",
                   horizontalalignment='center', fontsize=fontsize)
    i = i + 1
    ax_legend.text(.5, i*line_spacing, "Red: delta top of hill",
                   horizontalalignment='center', fontsize=fontsize)


def set_title(args):
    nfiles = len(args.input)
    titles = []
    if len(args.title) == 0:
        for item in args.input:
            titles.append(os.path.basename(item))
    else:
        if args.title[0] == "ETIKET":
            for item in args.input:
                titles.append("ETIKET")
        elif len(args.title) != nfiles:
            sys.exit("\nERROR with argument --title, expect " + str(nfiles)
                     + " value(s) but got "
                     + str(len(args.title)) + "\n")
        else:
            for item in args.title:
                titles.append(item)
    return titles

def check_legend(args, i, line_spacing):
    if (i+.05)*line_spacing > 1:
        raise RuntimeError("\n\nSome legend items do not fit in plot area.\n"
                           + "Please reduce the line spacing with argument "
                           + "\n\t--legend_text_spacing, the current "
                           + "value is " + str(args.legend_text_spacing)
                           + ".\nIf the lines are too close to each other, "
                           + "increase the legend plot area with argument "
                           + "\n\t--legend_fraction, current value is "
                           + str(args.legend_fraction) + ".\n\n")
