#!/usr/bin/perl
require 5.010_001;
#require 5.10.1;
use warnings;        # Avertissement des messages d'erreurs
use strict;          # Verification des declarations
use File::Spec::Functions;
#use File::Basename;
#use URI::file;
use Cwd qw( cwd realpath );
use Getopt::Long;

#########################################################################
# Process command line arguments
#########################################################################
my %msgl = ("q", 0 , "e", 1 ,"w", 2 ,"v", 3 ,"vv", 4, "vvv", 5 ) ;
my $help = 0;
my $msg =  2;
my $dup_ok = 0;
my $output_file='';
my $recurse = 0;
my @dirlist = ();
GetOptions('help'   => \$help,
			  'verbose:+' => \$msg,
           'quiet'  => sub{$msg=0;},
           'out=s'  => \$output_file,
			  'recurse' => \$recurse,
    )
    or $help=1;

@dirlist = @ARGV if ($#ARGV >= 0);

if (!$help and !($#dirlist+1)) {
    print STDERR "\nWARNING: no srcdir provided, using CWD\n\n" if ($msg >= 3);
	 @dirlist = (cwd());
}
if ($help) {
    print STDOUT "
Build source file dependencies list, GMakefile format
";
    exit;
}

#########################################################################
#########################################################################
my %TYPE_LIST = (
        f     => "COMPILABLE",
        ftn   => "COMPILABLE",
        ptn   => "", #INCLUDE,
        f90   => "COMPILABLE",
        f95   => "COMPILABLE",
        f03   => "COMPILABLE",
        ftn90 => "COMPILABLE",
        ptn90 => "", #INCLUDE,
        cdk   => "", #INCLUDE,
        cdk90 => "COMPILABLE",
        c     => "COMPILABLE",
        # cc    => "COMPILABLE",
        # cpp    => "COMPILABLE",
        h     => "", #INCLUDE,
        hf    => "", #INCLUDE,
        fh    => "", #INCLUDE,
        inc   => "", #INCLUDE,
        tmpl90 => "COMPILABLE",
    );

{ package SRCFile ; {

    # @new: Constructor of the SRCFile Object
    # IN :
    #   $1 = {
    #           path => 
    #           filename =>
    #           extension =>
    #        }
    # OUT: pointer to the object
    sub new {
        my ( $class, $ref_arguments ) = @_;
        
        $class = ref($class) || $class;
        my $this = {};
        bless( $this, $class );
        $this->{FULLPATH_SRC}     = " ";
        $this->{FILENAME}         = " ";
        $this->{EXTENSION}        = " ";
        $this->{FULLPATH_SRC}     = $ref_arguments->{path}; 
        $this->{FILENAME}         = $ref_arguments->{filename};
        $this->{EXTENSION}        = $ref_arguments->{extension};
        $this->{PATHyNAME}        = "$this->{FULLPATH_SRC}$this->{FILENAME}";
        $this->{NAMEyEXT}         = "$this->{FILENAME}.$this->{EXTENSION}";
        $this->{PATHyNAMEyEXT}    = "$this->{FULLPATH_SRC}$this->{NAMEyEXT}";
      %{$this->{DEPENDENCIES}}    = ();   # A list to the required file
        $this->{COMPILABLE}       = $TYPE_LIST{lc $this->{EXTENSION}};
        $this->{STATUS}           = undef;
      @{$this->{UNSOLVED_INC}} = ();
      @{$this->{UNSOLVED_USE}} = ();
      @{$this->{MODULE_LIST}}     = ();
      %{$this->{UNKNOWN_MODULE}}  = ();
      %{$this->{UNKOWN_USE}}      = ();
        return $this;
	 }

}} #end package SRCFile

#------------------------------------------------------------------------
# @pre_process_dir: Pre-Process src dir
#------------------------------------------------------------------------
sub pre_process_dir {
	 for (@dirlist){
		  if (! -d $_) {
				print STDERR "ERROR: Ignoring - Not a dir '$_' \n" if ($msg >= 3);
		  } else {
				print STDOUT "Pre-process_dir: '$_' $dup_ok\n" if ($msg >= 3);
            for (glob "$_/*") {
					 pre_process_file($_);
				}
		  }
	 }
}

#------------------------------------------------------------------------
# @pre_process_file: Pre-Process/register src file
# IN : 
#   $0 = file
#   $1 = ==1 if duplicatedfile_ok
# OUT: undef if ok; 1 otherwise
#------------------------------------------------------------------------
sub pre_process_file {
    my $file = $_[0] ;
    $file =~ s/[\s]+// ;
    $file =~ s/,v$// ;
	 $file = canonpath($file);

    return 1 if ($file !~  /(.*\/)*(.*)[.]([^.]*$)/);
    #return 1 if (exists($LISTOBJECT{$file})); 
    
    my $path = ($1 ? $1 : "");
    my $filn = ($2 ? $2 : "");
    my $exte = ($3 ? $3 : "");
    
    return 1 if (!exists($TYPE_LIST{lc $exte}));

	 print STDOUT "Pre-Process_file: '$file' $dup_ok\n" if ($msg >= 3);

	 my $keyname = "$filn.$exte"; #TODO: keyname could also be with path
	 if (exists($LISTOBJECT{$keyname})) {
		  #TODO: allow a mode for file override/replacement
		  print STDERR "ERROR: Ignoring - duplicate file: $filn.$exte ($path) \n" if ($msg >= 3);
		  return 1;
	 }
	 $LISTOBJECT{$keyname} = new SRCFile({path => $path, filename => $filn, extension => $exte})
}


#------------------------------------------------------------------------
# @process_all: Pro-process/parse all files
#------------------------------------------------------------------------
sub process_all {
	 for (keys(%LISTOBJECT)){
		  process_file($_);
	 }
}

#------------------------------------------------------------------------
# @process_file: Process/parse a src file
# IN : $keyname,$path,$filn,$exte
# OUT: undef if ok; 1 otherwise
#------------------------------------------------------------------------
sub process_file {
	 my ($keyname) = @_;
	 my $fileobj = $LISTOBJECT{$keyname};
    my $filepath = $fileobj->{PATHyNAMEyEXT};
    print STDERR "Looking into $filepath\n" if ($msg >= 3);
    open(INPUT,"<", $filepath) or print STDERR "\nERROR: Can't open file '".$filepath."\n"; #  if ($msg >= 1 )
    # my $line_number = 0;

    while (<INPUT>) {
        # $line_number++;
        if ($_ =~ /^[@]*[\s]*#[\s]*include[\s]*[<'"\s]([\w.\/\.]+)[>"'\s][\s]*/) {
				push @{$fileobj->{UNSOLVED_INC}}, $1; 
            next;# if (process_file_for_include($file,$1));
        }
        next if ($fileobj->{EXTENSION} =~ /^(c|cc|CC)$/);
        
        # FORTRAN include statement : include "..."    include ',,,"
        if ($_ =~ /^[@]*[\s]*include[\s]*[<'"\s]([\w.\/\.]+)[>"'\s][\s]*/i) {
				push @{$fileobj->{UNSOLVED_INC}}, $1; 
            next;# if (process_file_for_include($file,$1));
        }
        # FORTRAN use statement : use yyy 
        if ($_ =~ /^[@]*[\s]*\buse[\s]+([a-z][\w]*)(,|\t| |$)/i) {
            my $modname = $1 ; $modname =~ tr/A-Z/a-z/ ; # modules names are case insensitive
				push @{$fileobj->{UNSOLVED_USE}}, $modname; 
				next;
        } elsif ($_ =~ /^[@]*[\s]*\buse[\s]+([a-z][\w]*)/i) {
            #${$file->{UNKOWN_USE}}{$line_number} = $_;
            #print STDERR "$filepath ? \n";
            my $modname = $1 ; $modname =~ tr/A-Z/a-z/ ; # modules names are case insensitive
				push @{$fileobj->{UNSOLVED_USE}}, $modname; 
				next;
       }

        # FORTRAN module declaration : module xxx
        if ($_ =~ /^[@]*[\s]*\bmodule[\s]+([a-z][\w]*)(,|\t| |$)/i) {
            my $modname = $1 ; $modname =~ tr/A-Z/a-z/ ; # modules names are case insensitive
            next if $modname eq "procedure";
 				push @{$fileobj->{MODULE_LIST}}, $modname; 
        } elsif ($_ =~ /^[@]*[\s]*\bmodule[\s]+/i) {
            print STDERR "Unknown module statement: $filepath: $_\n";
				#TODO: save a list of Unknown module statements
       }
    }
    close INPUT;
}

#------------------------------------------------------------------------
# @post_process_all: post-process all files
#------------------------------------------------------------------------
sub post_process_all {
	 for (keys(%LISTOBJECT)){
		  post_process_file($_);
	 }
}

#------------------------------------------------------------------------
# @post_process_file: post-process a src file to recursively resolve dependencies
# IN : $keyname,$path,$filn,$exte
# OUT: undef if ok; 1 otherwise
#------------------------------------------------------------------------
sub post_process_file {
	 my ($keyname) = @_;
	 my $fileobj = $LISTOBJECT{$keyname};
}

#########################################################################
#########################################################################
my %LISTOBJECT = ( );
pre_process_dir();
process_all();
post_process_all();
