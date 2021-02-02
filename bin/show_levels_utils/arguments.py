import argparse


def show_levels_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        nargs='+',
        default="",
        help="Standard RPN fst file list containing the !! to plot levels for")
    parser.add_argument(
        "-o", "--out",
        default="",
        help="Output file name, default no figure")
    parser.add_argument(
        "-t", "--title",
        nargs='+',
        default="",
        help="Plot title, one by input file, default to file_name. The token"
        + "ETIKET may be used to write the ETIKET as title")
    parser.add_argument(
        "--pbot_hill",
        type=float,
        default=101325.,
        help="Pressure below the hill in hPa")
    parser.add_argument(
        "--ptop_hill",
        type=float,
        default=50000.,
        help="Pressure on top of the hill in hPa")
    parser.add_argument(
        "--small_scale_topo_amp",
        type=float,
        default=40000.,
        help="Amplitude for small scale topographic features in Pa.")
    parser.add_argument(
        "--small_scale_topo_cycles",
        type=float,
        default=4.,
        help="Number of sinus cycle in the scale topographic features "
        + "through the domain")
    parser.add_argument(
        "--plot_thermo",
        action="store_true",
        help='Plot thermodynamic levels')
    parser.add_argument(
        "--plot_vertical_velocity",
        action="store_true",
        help='Plot vertical velocity levels')
    parser.add_argument(
        "--ptop_graph",
        type=float,
        help="Pressure at graph top in hPa")
    parser.add_argument(
        "--pbot_graph",
        type=float,
        help="Pressure at graph bottom in hPa")
    parser.add_argument(
        "--plot_width",
        type=float,
        default=4.,
        help="Plot width")
    parser.add_argument(
        "--plot_height",
        type=int,
        default=10,
        help="Plot height")
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=90,
        help="Plot dpi")
    parser.add_argument(
        "--scale_fonts",
        type=float,
        default=1.,
        help="Apply a scalling factor on font size")
    parser.add_argument(
        "--allow_sigma",
        action="store_true",
        help="Allow ploting sigma coordinate. It is not permitted be default"
        + "to avoid potential error du to missing records in file e.g. PT"
        + "which can lead to erronious coordinate construction")
    parser.add_argument(
        "--vcode",
        action="store_true",
        help="Print Vcode in graph legend")
    parser.add_argument(
        "--plot_delta",
        action="store_true",
        help="Plot graph of layer thickness beside level graph")
    parser.add_argument(
        "--hide_diag_levels",
        action="store_true",
        help="Hide diag levels")    
    parser.add_argument(
        "--plot_delta_symb",
        action="store_true",
        help="Plot symboles on graph of layer thickness beside level graph")
    parser.add_argument(
        "--legend_fraction",
        type=float,
        default=.2,
        help="Fraction of the vertical space occupied by legend, typpical"
        + " value is .22")
    parser.add_argument(
        "--legend_text_spacing",
        type=float,
        default=.13,
        help="Fraction of the vertical legend space occupied by one line,"
        + " typpical value is .13")
    
    return parser.parse_args()
