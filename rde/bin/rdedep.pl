#!/usr/bin/perl
#use 5.010_001;
use 5.008_008;
#use warnings;        # Avertissement des messages d'erreurs
#use strict;          # Verification des declarations
use File::Spec::Functions;
use File::Basename;
use URI::file;
use Cwd "realpath";
use Getopt::Long;
my $msg =  2;
my %msgl = ("q", 0 , "e", 1 ,"w", 2 ,"v", 3 ,"vv", 4, "vvv", 5 ) ;
my $items_per_line = 4 ;   # number of items per Makefile line
my $item_count = 0;
my $ext = undef;
#my @listfile = ();
my %listfile = ();
my @includelist = ();
my $use_strict = undef;
my $deep_include = undef;
my $soft_restriction = undef; # soft_restriction = disable warning if 2 headers files have the same name
my $flat_layout = undef;
my $short_target_names = undef;
my $local_dir = undef;
my $dup_ok = undef;
my $side_dir_inc = undef;
my $anywhere_inc = undef;
my $export_list = undef;
#my @current_dependencies_list = ();
my %current_dependencies_list = ();
my @outside_tree_list = ();
my %module_missing = ();
my %module_dup = ();
my %module_list = ();
my %module_missing_ignored = ();
my %include_missing_ignored = ();
my %LISTOBJECT = ( ); # Hash of SRCFile object with filename, path and extension as the key
my %override_files = ();
my %dirdeplist = ();
my %invdeplist = ();
my %topdirnames = ();
my $myname  = "rdedep.pl";

#TODO: replace @arrays by %hash or $hash_ref ; %hash = map { $_ => 1 } @array;
#TODO: replace %hash by $hash_ref

#########################################################################
# Function for older perl versions
#########################################################################
sub mysmart {
   my $item = shift;
   my $myarray_ref = shift;
   # return ($item ~~ @$myarray_ref); #perl version >= 5.10.1 only
   return 0+ grep { $_ eq $item } @$myarray_ref;

   # use List::MoreUtils qw/any/
   # my $found = any { /\bsomething\b/      } @array_of_strings;
   # my $found = any { $_ == 42             } @array_of_numbers;
   # my $found = any { $_->isa('obj_type')  } @array;
   # my $found = any { $_->can('tupitar')   } @array;
}

#########################################################################
# Process command line arguments
#########################################################################
my $help = 0;
my $output_file='';
my $include_dirs='';
my $top_dirs='';
my $suppress_errors_file='';
my $defaultlibname = 'all';
my $defaultlibext = '.a';
my $override_dir='';
my $files_from='';
my $allow_circular_include=0; #TODO: add as option
GetOptions('help'   => \$help,
			  'verbose:+' => \$msg,
           'quiet'  => sub{$msg=0;},
           'supp=s' => \$suppress_errors_file,
           'exp=s'  => \$export_list,
           'out=s'  => \$output_file,
           'side_dir_inc' => \$side_dir_inc,
           'any_inc' => \$anywhere_inc,
           'includes=s' => \$include_dirs,
           # 'local' => \$local_dir,
			  'topdirs=s' => \$top_dirs,
           'strict' => \$use_strict,
           'deep-include' => \$deep_include,
           'soft-restriction' => \$soft_restriction,
           # 'dup_ok' => \$dup_ok,
           'flat_layout' => \$flat_layout,
           'short_target_names' => \$short_target_names,
			  'libext=s' => \$defaultlibext,
			  'override_dir=s' => \$override_dir,
			  'files_from=s' => \$files_from,
           # 'lib_target' => \$lib_target,
    )
    or $help=1;

#@listfile = @ARGV if ($#ARGV >= 0);
%listfile = map { $_ => 1 } @ARGV if ($#ARGV >= 0);

if ($files_from) {
   if (!$help and (keys %listfile)>=1) {
      $help=1;
      print STDERR "\nERROR: you cannot provied both --files_from and a list of files_dirs\n"
   }
   if (!open(INPUT,"<", $files_from)) {
      print STDERR "\nERROR: Can't open files_from file, ignoring: ".$files_from."\n";
   } else {
      while (<INPUT>) {
         $k = $_;
         chomp($k);
         if ($k) {
            $listfile{$k} = 1;
         }
      }
   }
}

if (!$help and (keys %listfile)<1) {
   $help=1;
   print STDERR "\nERROR: you must provide a list of targets\n"
}
if ($help) {
   print STDERR "
Build source file dependencies list, GMakefile format

Usage: $myname [-v|--quiet] \\
               [--supp=suppress_errors_file]  \\ 
               [--exp=output_of_produced_file] [--out=outfile] \\
               [--includes=list_of_inc_dirs]  \\ 
               [--side_dir_inc] [--any_inc] [--topdirs=list_of_top_dirs]\\
               [--strict] [--deep-include] [--soft-restriction] \\
               [--flat_layout] [--short_target_names] \\
               [--files_from=filelistfile] list_of_files_dirs
Options:
    -v     : verbose mode (multipe time to increaselevel)
   --quiet : no printed message
   --supp  : suppress_errors_file contains known error message to not print
   --exp   : Output the list of dependencies to the output file
   --out   : Output dependency rules to the output file [Default: STDOUT]

   --includes         : Search for include file in the listed dir, do not process these dir
   --any_inc          : add all dir in list_of_files_dirs to the include_path
   --side_dir_inc     : 
   --strict           : Error on include 'myfile' if myfile is compilable
   --deep-include     : Deep search for dependencies, recusive include
   --soft-restriction : Allow to include files to have the same name

   --flat_layout        : remove dir part in dependencies list
   --short_target_names : add PHONY obj target without path as
                          filename.o: path/filename.o
   --topdirs            : make list of files (among analysed files) for 
                          each top dir [colon separated]
                          (define OBJECTS_name, FORTRAN_MODULES_name vars)
                          
   --override_dir       : dir with locally modified source, 
                          overrides other sources files with same name
   --files_from         : File that includes the full list of 
                          files or dirs to process
list_of_files_dirs      : full list of files or dirs to process

suppress_errors_file sample:
   module_missing iso_c_binding
   include_missing model_macros_f.h
\n\n";

   exit;
}

print STDERR "
$myname \\
    --supp=$suppress_errors_file \\
    --exp=$export_list \\
    --out=$output_file \\
    --includes=$include_dirs \\ 
    --topdirs=$top_dirs \\
    --override_dir=$override_dir \\
    --side_dir_inc=$side_dir_inc --any_inc=$anywhere_inc  -v=$msg \\
    --strict=$use_stric       --deep-include=$deep_include --soft-restriction=$soft_restriction \\
    --flat_layout=$flat_layout --short_target_names=$short_target_names \\
    --files_from=$files_from \\
    ",join(" ", keys %listfile),"
   \n" if ($msg>=3);
#    ",join(" ", @listfile),"

#########################################################################
# List of function and object definition
#########################################################################

{ package SRCFile ; {
   # List of the type of file associated with the extension
   our %TYPE_LIST = (
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
      h     => "", #INCLUDE,
      hf    => "", #INCLUDE,
      fh    => "", #INCLUDE,
      inc   => "", #INCLUDE,
      include => "", #INCLUDE,
      tmpl90 => "COMPILABLE",
       );

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
      @{$this->{UNSOLVED_MODULE}} = ();
      @{$this->{MODULE_LIST}}     = ();
      %{$this->{UNKNOWN_MODULE}}  = ();
      %{$this->{UNKOWN_USE}}      = ();
      return $this;
   }

   # @has_module: find if the object defined the module 
   # IN : $1 = Module name to find
   # OUT: true (1) if the module as been found, false (undef) otherwise
   sub has_module {
      my $module_name = $_[1];    
      for (@{$_[0]->{MODULE_LIST}}) {
         return 1 if $_ eq $module_name;
      }
      return undef;
   }

   # @has_unsolved_module: find if the object has the module in his unsolved module list
   # IN : $1 = Module name to find
   # OUT: true (1) if the module as been found, false (undef) otherwise
   sub has_unsolved_module {
      my $module_name = $_[1];        
      for (@{$_[0]->{UNSOLVED_MODULE}}) {
         return 1 if ($_ eq $module_name);
      }
      return undef;
   }

   # @remove_unsolved_module: delete the module in the unsolved module list
   # IN : $1 = Module name to delete
   # OUT: none
   sub remove_unsolved_module {
      my $module_name = "$_[1]";
      my $idx = 0;
      for my $module (@{$_[0]->{UNSOLVED_MODULE}}) {
         if ($module eq $module_name) { 
            delete ${$_[0]->{UNSOLVED_MODULE}}[$idx]; 
         } 
         $idx++; 
      }
   }

   # @find_dependencies: find if the object has the filename in his depedencie list
   # IN : $1 = filename to search
   # OUT: true (1) if the module as been found, false (undef) otherwise
   sub find_dependencies {
      my $search_depedencies = "$_[1]";
      #print STDERR "find_dependencies: $_[1]:$_[0]:",keys %{$_[0]->{DEPENDENCIES}},"\n";
      #print STDERR "find_dependencies: $search_depedencies in dep of ",$_[0]->{NAMEyEXT},": ndep=",int(keys %{$_[0]->{DEPENDENCIES}}),"\n";
      for my $dep_filename (keys %{$_[0]->{DEPENDENCIES}}) {
         #print STDERR "find_dependencies: $search_depedencies ?=? $dep_filename\n";
         my $dep_ptr = ${$_[0]->{DEPENDENCIES}}{$dep_filename};
         if ($flat_layout) {
            return 1 if ($search_depedencies eq $dep_filename);
         } else {
            return 1 if ($search_depedencies eq $dep_filename and $_[0]->{FULLPATH_SRC} eq $dep_ptr->{FULLPATH_SRC});
         }
         return 1 if ($dep_ptr->SRCFile::find_dependencies($search_depedencies));
      }
      return undef;
   }

  }} #end package SRCFile

#------------------------------------------------------------------------
# @preproc_suppfile: Process suppress_errors_file
#------------------------------------------------------------------------
sub preproc_suppfile {
   my $suppfile = shift;
   if ($suppfile) {
      print STDERR "Processing suppress_errors_file: $suppfile\n" if ($msg >= 3);
      if (!open(INPUT,"<", $suppfile)) {
         print STDERR "\nERROR: Can't open supp file, ignoring: ".$suppfile."\n";
      } else {
         while (<INPUT>) {
            if ($_ =~ /^[\s]*module_missing[\s]+([^\s]+)/i) {
               print STDERR "Suppressing missing mod msg for: ".$1."\n" if ($msg >= 4);
               $module_missing_ignored{$1} = 1;
            } elsif ($_ =~ /^[\s]*include_missing[\s]+([^\s]+)/i) {
               print STDERR "Suppressing missing inc msg for: ".$1."\n" if ($msg >= 4);
               $include_missing_ignored{$1} = 1;
            } else {
               print STDERR "Ignoring supp file line: ".$_."\n" if ($msg >= 4);
            }
         }
         close INPUT;
      }
   }
   return undef;
}

#------------------------------------------------------------------------
# @preproc_srcfiles: Pre-Process src file
#------------------------------------------------------------------------
sub preproc_srcfiles_overrides {
   return if !$override_dir;
   for (glob "$override_dir/*") {		  
      my $file0 = "$_" ;
      my $file = "$_" ;
      $file =~ s/,v$// ;
      $file =~ s/[\s]+// ;
      #$file = File::Spec->abs2rel(canonpath($file), "./") if ($file =~ /^\//);
      $file = canonpath($file);
      if ($file =~  /(.*\/)*(.*)[.]([^.]*$)/) {
         my $filn = ($2 ? $2 : "");
         my $exte = ($3 ? $3 : "");
         if (has_legal_extension($exte)) {
            #push @override_files, "$filn.$exte";
            $override_files{"$filn.$exte"} = $file0;
         }
      }
   }
   if ($msg >= 3 and $#override_files) {
      print STDERR "override_files = ";
      for (keys %override_files) {
         print STDERR $_." ";
      }
      print STDERR "\n";
   }
}

sub preproc_srcfiles {
   #print STDERR "\npreproc_srcfiles $#listfile",@listfile,"\n";
   #for (@listfile){
   for (keys %listfile){
      if (-d $_) {
         print STDERR "Pre-processD: '$_' $dup_ok\n" if ($msg >= 3);
         for (glob "$_/*") {
            pre_process_file($_,$dup_ok,0);
         }
      } else {
         print STDERR "Pre-processF: '$_'\n" if ($msg >= 3);
         for (glob "$_") {
            pre_process_file($_,$dup_ok,0);
         }
      }
   }

   $dup_ok = 1;
   if ($local_dir) {
      print STDERR "Pre-processL: Local dir\n" if ($msg >= 3);
      for (glob "./*") {
         pre_process_file($_,$dup_ok);
      }
   }

   for (keys %override_files) {
      print STDERR "Pre-processO: '$_' ($override_files{$_})\n" if ($msg >= 3);
      pre_process_file($override_files{$_},$dup_ok,1);
   }
}

#------------------------------------------------------------------------
# @process_file
# IN : 
#   $0 = file
#   $1 = ==1 if duplicatedfile_ok
# OUT: undef if ok; 1 otherwise
#------------------------------------------------------------------------
sub pre_process_file {
   return 1 if (! -f "$_[0]");
   my $entry = shift;
   my $_dup_ok = shift;
   my $_is_override = shift;

   my $file = "$entry" ;
   $file =~ s/,v$// ;
   $file =~ s/[\s]+// ;
   #$file = File::Spec->abs2rel(canonpath($file), "./") if ($file =~ /^\//);
   $file = canonpath($file);

   return 1 if ($file !~  /(.*\/)*(.*)[.]([^.]*$)/);
   #return 1 if (exists($LISTOBJECT{$file})); 
   
   my $path = ($1 ? $1 : "");
   my $filn = ($2 ? $2 : "");
   my $exte = ($3 ? $3 : "");
   
   return 1 if (!has_legal_extension($exte));
   #return 0 if (exists($override_files{"$filn.$exte"}));

   my $duplicated_filename1 = find_same_filename2($path, "$filn.$exte");

   #print STDERR "pre_process_file: $filn.$exte ($path) ($duplicated_filename1)\n";

   if ($duplicated_filename1 and ($_dup_ok or $_is_override)) {
      #print STDERR "DELETE dup: $duplicated_filename1 (for $path$filn.$exte)\n";
      delete $LISTOBJECT{$duplicated_filename1};
   }

   #print STDERR "NEW1: $filn.$exte ($path)\n";
   $LISTOBJECT{"$path$filn.$exte"} = new SRCFile({path => $path, filename => $filn, extension => $exte});

   if (!$_is_override) {
      if ($duplicated_filename1 and $_dup_ok) {
         print STDERR "\nWARNING: $duplicated_filename1 was replaced by $path$filn.$exte : ".$LISTOBJECT{"$path$filn.$exte"}->{FILENAME}.$LISTOBJECT{"$path$filn.$exte"}->{STATUS};
      }
      
      # Error handler
      my $duplicated_filename2 = find_same_output("$path$filn.$exte");
      if ($_dup_ok) {
         if ($msg >= 1) {
            print STDERR "\nWARNING: using 2 files with the same name $duplicated_filename1 with $path$filn.$exte\n" if ($duplicated_filename1);
            print STDERR  "\nWARNING: using 2 files ($duplicated_filename2 and $path$filn.$exte) that will produce the same object file ($filn.o)\n" if ($duplicated_filename2);
         }
      } else {
         die "\nERROR1: using 2 files with the same name $duplicated_filename1 with $path$filn.$exte" if ($duplicated_filename1);
         die "\nERROR2: using 2 files ($duplicated_filename2 and $path$filn.$exte) that will produce the same object file ($filn.o)\n" if ($duplicated_filename2);        
      }
   } else {
      my $duplicated_output2 = find_same_output2("$path$filn.$exte");
      if ($_dup_ok) {
         print STDERR  "\nWARNING: using 2 files ($duplicated_output2 and $path$filn.$exte) that will produce the same object file ($filn.o)\n\n" if ($duplicated_output2);
      } else {
         die "\nERROR3: using 2 files ($duplicated_output2 and $path$filn.$exte) that will produce the same object file ($filn.o)\n" if ($duplicated_output2);
      }
   }
   # print STDERR "process: '$entry' dupok=$_dup_ok ; path=$path ; filen=$filn ; exte=$exte ; dup=$duplicated_filename1\n" if ($msg >= 5);
   return undef;
}

#------------------------------------------------------------------------
# @find_same_filename: Look if the filename is already used somewhere else
# IN : $0 = filename to compare with
# OUT: object key (filename) if file already exist, false (undef) otherwise
#------------------------------------------------------------------------
sub find_same_filename {
   my $cmp_file = $LISTOBJECT{$_[0]};
   return undef if ($soft_restriction and !$cmp_file->{COMPILABLE});
   for (keys %LISTOBJECT) {
      if (
         ($LISTOBJECT{$_}->{NAMEyEXT} eq $cmp_file->{NAMEyEXT}) and
         ($LISTOBJECT{$_}->{FULLPATH_SRC} ne $cmp_file->{FULLPATH_SRC})) {
         print STDERR "\n$LISTOBJECT{$_}->{NAMEyEXT}: $LISTOBJECT{$_}->{FULLPATH_SRC} != $cmp_file->{FULLPATH_SRC}\n"; 
      }
      
      return $_ if (
         ($LISTOBJECT{$_}->{NAMEyEXT} eq $cmp_file->{NAMEyEXT}) and
         ($LISTOBJECT{$_}->{FULLPATH_SRC} ne $cmp_file->{FULLPATH_SRC}));
   }
   return undef;
}

#------------------------------------------------------------------------
# @find_same_filename: Look if the filename is already used somewhere else
# IN : 
#   $0 = path to filename to compare with
#   $1 = filename.ext to compare with
# OUT: object key (filename) if file already exist, false (undef) otherwise
#------------------------------------------------------------------------
sub find_same_filename2 {
   my ($mydir, $myfilename) = @_;
   for (keys %LISTOBJECT) {
      if (($LISTOBJECT{$_}->{NAMEyEXT} eq $myfilename) and
          ($LISTOBJECT{$_}->{FULLPATH_SRC} ne $mydir)) {
         return undef if ($soft_restriction and !$LISTOBJECT{$_}->{COMPILABLE});
         return $_
      }
   }
   return undef;
}

#------------------------------------------------------------------------
# @find_same_OUT: Look if the filename is already used in the Object list
# IN : $0 = Object to compare with
# OUT: object key (filename) if file already exist, false (undef) otherwise
#------------------------------------------------------------------------
sub find_same_output {
   my $cmp_file = $LISTOBJECT{$_[0]};
   return undef if (!$cmp_file->{COMPILABLE});
   for my $key (keys %LISTOBJECT) {
      return $key if (
         ($LISTOBJECT{$key}->{FILENAME} eq $cmp_file->{FILENAME}) and 
         $LISTOBJECT{$key}->{COMPILABLE} and
         !($LISTOBJECT{$key}->{FULLPATH_SRC} eq $cmp_file->{FULLPATH_SRC}) and 
         ($key ne $_[1])); 
   }
   return undef;
}

sub find_same_output2 {
   my $cmp_file = $LISTOBJECT{$_[0]};
   return undef if (!$cmp_file->{COMPILABLE});
   for my $key (keys %LISTOBJECT) {
      return $key if (
         ($LISTOBJECT{$key}->{FILENAME} eq $cmp_file->{FILENAME}) and 
         $LISTOBJECT{$key}->{COMPILABLE} and
         !($LISTOBJECT{$key}->{EXTENSION} eq $cmp_file->{EXTENSION}) and
         ($key ne $_[1])); 
   }
   return undef;
}

#------------------------------------------------------------------------
# @search_undone_file
#------------------------------------------------------------------------
sub search_undone_file {
   for (keys %LISTOBJECT) {
      return $_ if !$LISTOBJECT{$_}->{STATUS}; 
   }
   return undef;
}

#------------------------------------------------------------------------
# @process_file
# IN : 01 = filename
# OUT: undef if ok; 1 otherwise
#------------------------------------------------------------------------
sub process_file {
   my $filename = $_[0];
   print STDERR "Looking into $filename\n" if ($msg >= 5);
   open(INPUT,"<", $filename) or print STDERR "\nERROR: Can't open file '".$filename."\n"; #  if ($msg >= 1 )
   my $file = $LISTOBJECT{$filename};
   my $line_number = 0;

   while (<INPUT>) {
      if ($_ =~ /^[@]*[\s]*#[\s]*include[\s]*[<'"\s]([\w.\/\.]+)[>"'\s][\s]*/) {
         #print STDERR "sub_process_file0: $filename : include = $1\n";
         next if (process_file_for_include($file,$1));
      }
      next if ($file->{EXTENSION} =~ /^(c|cc|CC)$/);

      if ($file->{EXTENSION} =~ /^(F90)$/) {
         if ($_ =~ /^[\s\t]*![\s\t]*\/\*/) {
            print STDERR "\nWARNING: File $filename has C style comments (/* ... */)\n"
         }
      }

      # FORTRAN include statement : include "..."    include ',,,"
      if ($_ =~ /^[@]*[\s]*include[\s]*[<'"\s]([\w.\/\.]+)[>"'\s][\s]*/i) {
         #print STDERR "sub_process_file1: $filename : include = $1\n";
         next if (process_file_for_include($file,$1));
      }
      # FORTRAN use statement : use yyy 
      if ($_ =~ /^[@]*[\s]*\buse[\s]+([a-z][\w]*)(,|\t| |$)/i) {
         my $modname = $1 ; $modname =~ tr/A-Z/a-z/ ; # modules names are case insensitive

         # If the module can be found, add the file to dependencies
         if (my $include_filename = search_module($modname)) {
            #print STDERR "module $modname in file $include_filename used in ",$file->{PATHyNAMEyEXT},"\n";
            if ($include_filename ne $file->{PATHyNAMEyEXT}) {
               ${$file->{DEPENDENCIES}}{$include_filename} = $LISTOBJECT{$include_filename} if (!exists ${$file->{DEPENDENCIES}}{$include_filename} );
               #print STDERR "$filename +: $modname \n";
            }
         } else {
            push @{$file->{UNSOLVED_MODULE}}, $modname if (!mysmart($modname,\@{$file->{UNSOLVED_MODULE}}));
            #print STDERR "$filename -: $modname \n";
         }

      } elsif ($_ =~ /^[@]*[\s]*\buse[\s]+([a-z][\w]*)/i) {
         ${$file->{UNKOWN_USE}}{$line_number} = $_;
         #print STDERR "$filename ? \n";
      }

      # FORTRAN module declaration : module xxx
      if ($_ =~ /^[@]*[\s]*\bmodule[\s]+([a-z][\w]*)(,|\t| |$)/i) {
         my $modname = $1 ; $modname =~ tr/A-Z/a-z/ ; # modules names are case insensitive
         my $search_filename = '';

         next if $modname eq "procedure";

         # Verifier que le nom du module n'existe pas dans un autre fichier
         if ($search_filename = search_module($modname)) { 
            print STDERR "\n\nERROR: Multiple definitions of Module '".$modname."'\n";
            print STDERR "       1: ".$search_filename."\n"; 
            print STDERR "       2: ".$filename."\n"; 
            print STDERR "==== ABORT ====\n\n"; 
            close INPUT;
            unlink $output_file;
            exit 1;
            #TODO: better exception handeling
            # $module_dup{$modname} = () if not exists($module_dup{$modname});
            # %a = $module_dup{$modname};
            # $a{$search_filename} = 1 if not exists($a{$search_filename});
            # $a{$filename} = 1 if not exists($a{$filename});
            next; 
         }

         # Ajouter le module dans la liste des modules associer au fichier.
         push @{$file->{MODULE_LIST}}, $modname if (!mysmart($modname,\@{$file->{MODULE_LIST}}));
         if ($flat_layout) {
            $module_list{$modname} = $file->{NAMEyEXT};
         } else {
            $module_list{$modname} = $filename;
         }

         # Recherche tous les fichiers analyser precedemment qui avait besoin de ce module la
         while(my $key = search_unsolved_module($modname)) {
            #print STDERR "unsolved module: $key".${$LISTOBJECT{$key}->{ DEPENDENCIES }}{$filename}." : $modname \n";
            # Ajouter a la liste des dependence, le fichier en cours
            ${$LISTOBJECT{$key}->{ DEPENDENCIES }}{$filename} = $file if (!exists ${$LISTOBJECT{$key}->{ DEPENDENCIES }}{$filename} );

            # Enlever le module de la liste des unsolved modules 
            $LISTOBJECT{$key}->remove_unsolved_module($modname);
         }
      } elsif ($_ =~ /^[@]*[\s]*\bmodule[\s]+/i) {
         ${$file->{ UNKOWN_MODULE }}{$line_number} = $_;
         #print STDERR "Unknown module statement: $filename: $_\n";
      }
      $line_number++;
   }
   $file->{STATUS} = 1;
   close INPUT;
}

#------------------------------------------------------------------------
# @process_file_for_include
# IN : 
#   $0 = file object
#   $1 = filename
# OUT: undef if ok; 1 otherwise
#------------------------------------------------------------------------
sub process_file_for_include {
   my ($file, $tmp_dir) = @_;
   my $include_path = "";

   if ($tmp_dir =~ /^\.\.\//) {
      #$include_path = File::Spec->abs2rel(canonpath("$file->{FULLPATH_SRC}/$tmp_dir"), "./");
      $include_path = canonpath("$file->{FULLPATH_SRC}/$tmp_dir");
   } elsif (-f canonpath("$file->{FULLPATH_SRC}/$tmp_dir")) {
      #$include_path = File::Spec->abs2rel(canonpath("$file->{FULLPATH_SRC}/$tmp_dir"), "./");
      $include_path = canonpath("$file->{FULLPATH_SRC}/$tmp_dir");
   } else {
      #$include_path = File::Spec->abs2rel( canonpath($tmp_dir), "./");
      $include_path = canonpath("$tmp_dir");
   }
   # print STDERR "Missing $file->{NAMEyEXT}: $tmp_dir\n" if (!$include_path and $msg>=4);
   
   if ($include_path !~  /(.*\/)*(.*)[.]([^.]*$)/) {
      # print STDERR "Outside $file->{NAMEyEXT}: $tmp_dir : $include_path\n" if ($msg>=4);
      return 1;
   }

   #print STDERR "process_file_for_include: $file->{PATHyNAME} : $tmp_dir : $include_path\n";

   my $path = ($1 ? $1 : "");
   my $filn = ($2 ? $2 : "");
   my $exte = ($3 ? $3 : "");
   my $duplicated_filename = "";
   if (!has_legal_extension($exte)) {
      # print STDERR "Bad Extension $file->{NAMEyEXT}: $tmp_dir : $include_path : $exte\n" if ($msg>=4);
      return 1;
   }
   ## if (!exists $LISTOBJECT{"$path$filn.$exte"}) {
   if ((! -f "$path$filn.$exte") or ($anywhere_inc and !exists $LISTOBJECT{"$path$filn.$exte"})) {
      #if (!("$path$filn.$exte" ~~ @outside_tree_list)) {
      if (!mysmart("$path$filn.$exte",\@outside_tree_list)) {
         my $path1 = find_inc_file($file,$path,"$filn.$exte");
         if (!$path1) {
            #print STDERR "No file $file->{NAMEyEXT}: $tmp_dir : $include_path : $path$filn.$exte\n";# if ($msg>=4);
            if ("$path$filn.$exte" !~ @outside_tree_list and !exists($include_missing_ignored{"$path$filn.$exte"})) {
               push @outside_tree_list, "$path$filn.$exte";
            }
            return 1;
         }
         #print STDERR "Found $filn.$exte in $path1\n";# if ($msg >=5);
         $path = $path1;
      } else {
         return 1;
      }
   }
   ##}

   #Check if file is including itself
   #print STDERR "process_file_for_include: ",$file->{FULLPATH_SRC}," : ",$file->{PATHyNAMEyEXT}," : inc=$path$filn.$exte\n";
   if (!$allow_circular_include) {
      if ($file->{PATHyNAMEyEXT} eq "$path$filn.$exte") {
         die "\nERROR: File cannot include itself: $path$filn.$exte\n"
      }
   }

   # Add file in the database if it's not in yet and if the file really exists.
   if (!exists $LISTOBJECT{"$path$filn.$exte"}) {
      #print STDERR "NEW2: $filn.$exte ($path)\n";
      $LISTOBJECT{"$path$filn.$exte"} = new SRCFile({path => $path, filename => $filn, extension => $exte});
   }

   # Force the file to not be analysed.
   $LISTOBJECT{"$path$filn.$exte"}->{STATUS} = 1 
       if (!$deep_include);

   # Error handler
   die "\nERROR3: using 2 files with the same name $duplicated_filename with $path$filn.$exte\n" 
       if ($duplicated_filename = find_same_filename("$path$filn.$exte"));
   die "\nERROR4: using 2 files ($duplicated_filename and $path$filn.$exte) that will produce the same object file ($filn.o)\n" 
       if ($duplicated_filename = find_same_output("$path$filn.$exte"));
   die "\nERROR5: cannot include compilable file ($tmp_dir) in $tmp_dir while using strict mode\n" 
       if ($use_strict and $LISTOBJECT{"$path$filn.$exte"}->{COMPILABLE});

   # Add to dependencies, if not already there
   ${$file->{ DEPENDENCIES }}{"$path$filn.$exte"} = $LISTOBJECT{"$path$filn.$exte"} if (!exists ${$file->{ DEPENDENCIES }}{"$path$filn.$exte"});

   return undef;
}

#------------------------------------------------------------------------
# @has_legal_extension: 
# IN : $0 = Extension to search
# OUT: 1 if the extension is valid, undef otherwise.
#------------------------------------------------------------------------
sub has_legal_extension {
   # my $search_extension = lc $_[0];
   # for (keys(%SRCFile::TYPE_LIST)) {
   #     return 1 if $_ eq $search_extension;
   # }
   return 1 if (exists($SRCFile::TYPE_LIST{lc $_[0]}));
   return undef;
}

#------------------------------------------------------------------------
# @check_circular_dep
#------------------------------------------------------------------------
sub check_circular_dep {
   return if ($allow_circular_include);
   print STDERR "Checking for Circular dependencies\n" if ($msg >= 3);
   #print STDERR "check_circular_dep: nfiles=",int(keys %LISTOBJECT),"\n";
   for (keys %LISTOBJECT) {
      #print STDERR "check_circular_dep ",$LISTOBJECT{$_}->{FILENAME}," : $_ : dep=",keys %{$LISTOBJECT{$_}->{DEPENDENCIES}},"\n";
      if ($LISTOBJECT{$_}->find_dependencies($_)) { 
         die "\nERROR: Circular dependencies in $_ FAILED\n";
      }
   }
   print STDERR "Done Checking for Circular dependencies\n" if ($msg >= 3);
}

#------------------------------------------------------------------------
# @print_header: Print the first line of a dependency rule or variable list
# IN :
#   $0 = First word(s) of line
#   $1 = Seperator
#   $2 = Word/item right after seperator, empty if no word is needed
# OUT: none
#------------------------------------------------------------------------
sub print_header {
   my($item1,$separ,$item2) = @_ ;
   $item_count = 0 ;
   print STDOUT "$item1$separ" ;
   print STDOUT "\t$item2" if ( "$item1" ne "$item2" && "$item2" ne "" ) ;
}

#------------------------------------------------------------------------
# @print_item: print each item of dependency rule or variable list (items_per_line items per line)
# IN : $0 = Item to print
# OUT: none
#------------------------------------------------------------------------
sub print_item {
   if ($_[0]) {
      print STDOUT " \\\n\t" if ($item_count == 0) ;
      print STDOUT "$_[0]  ";
      $item_count = 0 if ($item_count++ >= $items_per_line);
   }
}

#------------------------------------------------------------------------
# @print_files_list
#------------------------------------------------------------------------
sub print_files_list{
   print STDERR "Listing file types FDECKS, CDECKS, ...\n" if ($msg >= 3);
   for $ext (keys %SRCFile::TYPE_LIST) {
      print_header(uc $ext."DECKS", "=", "");
      for (sort keys %LISTOBJECT) {
         my $file = $LISTOBJECT{$_};
         if (lc $file->{EXTENSION} eq lc $ext) {
            if ($flat_layout) {
               print_item($file->{NAMEyEXT});
            } else {
               print_item($file->{PATHyNAMEyEXT});
            }
         }
      }
      print STDOUT "\n";
   }
}

#------------------------------------------------------------------------
# @print_object_list
#------------------------------------------------------------------------
sub get_topname {
   my $topdir = shift;
   if (!exists($topdirnames{$topdir})) {
      my $topname = undef;
      my $fileok = 1;
      my $fh = undef;
      if (-f "$topdir/.name") {
         open($fh, '<', "$topdir/.name") or $fileok = undef;
      }
      if ($fileok) {
         local $/;
         $topname = <$fh>;
         chomp($topname);
         $topname =~ s/^\s+//;
         $topname =~ s/\s+$//;
         close($fh);
      } else {
         my(@dirs) = split("/",$topdir);
         $topname = $dirs[$#dirs];
         $topname = $dirs[$#dirs-1] if ($topname eq 'src');
      }
      $topdirnames{$topdir} = $topname;
   }
   return $topdirnames{$topdir};
}

sub print_object_list {
   my %listdir = ();
   my %listsubdir = ();
   my %listtopdir = ();
   print STDERR "Listing OBJECTS\n" if ($msg >= 3);
   print_header("OBJECTS","=","");
   for (sort keys %LISTOBJECT) {
      my $file = $LISTOBJECT{$_};
      if ($file->{COMPILABLE}) {
         if ($flat_layout) {
            print_item("$file->{FILENAME}.o");
         } else {
            print_item("$file->{PATHyNAME}.o");
         }
         my(@dirs) = split("/",$file->{FULLPATH_SRC});
         @{$listdir{$dirs[0]}} = () if (!exists($listdir{$dirs[0]}));
         if ($flat_layout) {
            push @{$listdir{$dirs[0]}},"$file->{FILENAME}.o" if (!mysmart("$file->{FILENAME}.o",\@{$listdir{$dirs[0]}}));
         } else {
            push @{$listdir{$dirs[0]}},"$file->{PATHyNAME}.o" if (!mysmart("$file->{PATHyNAME}.o",\@{$listdir{$dirs[0]}}));
         }
      }
      if ($top_dirs) {
         for (split(":",$top_dirs)) {
            my $topdir = $_;
            if ($_) {
               my $topname = get_topname($topdir);
               if (index($file->{FULLPATH_SRC}, $topdir) != -1) {
                  if ($file->{COMPILABLE}) {
                     @{$listdir{$topname}} = () if (!exists($listdir{$topname}));
                     if ($flat_layout) {
                        push @{$listdir{$topname}},"$file->{FILENAME}.o" if (!mysmart("$file->{FILENAME}.o",\@{$listdir{$topname}}));
                     } else {
                        push @{$listdir{$topname}},"$file->{PATHyNAME}.o" if (!mysmart("$file->{PATHyNAME}.o",\@{$listdir{$topname}}));
                     }
                     my $subdirname = substr($file->{FULLPATH_SRC},length($topdir),-1);
                     if (not $subdirname) {
                        $subdirname = '_';
                     } else {
                        $subdirname =~ s|^src/||;
                        $subdirname =~ s|^/src/|/|;
                        $subdirname =~ s|/|_|g;
                     }
                     $subdirname = $topname.$subdirname;
                     @{$listsubdir{$topname}} = () if (!exists($listsubdir{$topname}));
                     push @{$listsubdir{$topname}},$subdirname if (!mysmart($subdirname,\@{$listsubdir{$topname}}));

                     $listtopdir{$topname} = $topdir if (!exists($listtopdir{$topname}));
                     $listtopdir{$subdirname} = $file->{FULLPATH_SRC} if (!exists($listtopdir{$subdirname}));

                     if ($flat_layout) {
                        push @{$listdir{$subdirname}},"$file->{FILENAME}.o" if (!mysmart("$file->{FILENAME}.o",\@{$listdir{$subdirname}}));
                     } else {
                        push @{$listdir{$subdirname}},"$file->{PATHyNAME}.o" if (!mysmart("$file->{PATHyNAME}.o",\@{$listdir{$subdirname}}));
                     }
                  }
               }
            }
         }
      }
   }
   print STDOUT "\n";

   #TODO: this should be optional
   for (keys %listtopdir) {
      my $libname = $_;
      $libname = $defaultlibname if $libname eq '..' or $libname eq '.' or $libname eq '';
      print_header("DIRORIG_".$libname,"=","$listtopdir{$_}");
      print STDOUT "\n";
   }
   print STDOUT "\n";

   #TODO: this should be optional
   print_header("TOPDIRLIST","=",join(" ",split(":",$top_dirs)));
   print STDOUT "\n";
   print_header("TOPDIRLIST_NAMES","=","");
   for (split(":",$top_dirs)) {
      print_item(get_topname($_));
   }
   print STDOUT "\n";

   for (keys %listsubdir) {
      my $libname = $_;
      $libname = $defaultlibname if $libname eq '..' or $libname eq '.' or $libname eq '';
      print_header("SUBDIRLIST_".$libname,"=","");
      for my $item (@{$listsubdir{$_}}) {
         $item =~ s|^[^_]*_||;
         print_item("$item");
      }
      print STDOUT "\n";
   }
   for (keys %listdir) {
      my $libname = $_;
      $libname = $defaultlibname if $libname eq '..' or $libname eq '.' or $libname eq '';
      print_header("OBJECTS_".$libname,"=","");
      for my $item (@{$listdir{$_}}) {
         print_item("$item");
      }
      print STDOUT "\n";
      print STDOUT "\$(LIBDIR)/lib".$libname.$defaultlibext.": \$(OBJECTS_".$libname.") \$(LIBDEP_".$libname.") \$(LIBDEP_ALL)\n";
      print STDOUT "\t".'rm -f $@; ar r $@_$$$$ $(OBJECTS_'.$libname.'); mv $@_$$$$ $@'."\n";
      print STDOUT "lib".$libname.$defaultlibext.": \$(LIBDIR)/lib".$libname.$defaultlibext."\n";
   }
   print STDOUT "\n";
   print_header("ALL_LIBS=","");
   for (keys %listdir) {
      my $libname = $_;
      $libname = $defaultlibname if $libname eq '..' or $libname eq '.' or $libname eq '';
      print_item("\$(LIBDIR)/lib".$libname.$defaultlibext);
   }
   print STDOUT "\n\n";
}

#------------------------------------------------------------------------
# @print_module_list
#------------------------------------------------------------------------
sub print_module_list {
   my %listdir = ();
   print STDERR "Listing FORTRAN_MODULES\n" if ($msg >= 3);
   print_header("FORTRAN_MODULES","=","");
   for (sort keys %module_list) {
      print_item("$_");
   }
   print STDOUT "\n\n";
   for (sort keys %module_list) {
      #TODO: if we know the module name we should make a dependency
      print STDOUT "FMOD_FILE_$_ = $module_list{$_}\n";
   }
   print STDOUT "\n";
   for (sort keys %LISTOBJECT) {
      my $file = $LISTOBJECT{$_};
      my @list_of_modules = ();
      for (@{$file->{MODULE_LIST}}) {
         push @list_of_modules, $_ if $_ ne "" and (!mysmart($_,\@list_of_modules));
      }
      if ($#list_of_modules >= 0) {
         print_header("FMOD_LIST_$file->{NAMEyEXT}","=","");
         for (@list_of_modules) {
            print_item("$_");
         }
         print STDOUT "\n";

         if ($top_dirs) {
            for (split(":",$top_dirs)) {
               my $topdir = $_;
               if ($_) {
                  my $topname = get_topname($topdir);
                  if (index($file->{FULLPATH_SRC}, $topdir) != -1) {
                     @{$listdir{$topname}} = () if (!exists($listdir{$topname}));
                     for (@list_of_modules) {
                        push @{$listdir{$topname}},"$_" if (!mysmart("$_",\@{$listdir{$topname}}));
                     }
                  }
               }
            }
         }
      }
   }
   print STDOUT "\n";
   
   #TODO: this should be optional
   for (keys %listdir) {
      my $libname = $_;
      $libname = $defaultlibname if $libname eq '..' or $libname eq '.' or $libname eq '';
      print_header("FORTRAN_MODULES_".$libname,"=","");
      for my $item (@{$listdir{$_}}) {
         print_item("$item");
      }
      print STDOUT "\n\n";
   }
}

#------------------------------------------------------------------------
# @print_targets
#------------------------------------------------------------------------
sub print_targets {
   print STDERR "Add custom targets\n" if ($msg >= 3);
   print STDOUT "\n";
   print STDOUT '$(eval MYVAR2 = $$($(MYVAR)))'."\n";
   print STDOUT "echo_mydepvar:\n";
   print STDOUT "\t".'echo $(MYVAR2)'."\n";
   print STDOUT "\n";
}

#------------------------------------------------------------------------
# @print_dep_rules
#------------------------------------------------------------------------
sub print_dep_rules {
   #TODO: Dependencies to Modules should be on .mod:.o not direcly on .o (.mod could have been erased accidentaly)
   print STDERR "Printing dependency rules\n" if ($msg >= 3);
   print STDOUT "\n";
   for my $filename (sort keys %LISTOBJECT) {
      my $file = $LISTOBJECT{$filename};
      #@current_dependencies_list = ();
      %current_dependencies_list = ();
      if ($file->{COMPILABLE}) {
         my $ext2 = $file->{EXTENSION};
         $ext2 = 'f'   if ($file->{EXTENSION} eq 'ftn');
         $ext2 = 'f90' if ($file->{EXTENSION} eq 'ftn90' or $file->{EXTENSION} eq 'cdk90');
         if ($flat_layout) {
            print_header("$file->{FILENAME}.o",":","$file->{FILENAME}.$ext2");
         } else {
            print_header("$file->{PATHyNAME}.o",":","$file->{PATHyNAME}.$ext2");
         }
         rec_print_dependencies(\%LISTOBJECT, $filename);
         #print STDOUT "\n";
         print_header("$file->{FILENAME}.o",":","$file->{PATHyNAME}.o") if ($short_target_names and $file->{FULLPATH_SRC} and !$flat_layout);
         print STDOUT "\n";

         #@current_dependencies_list = ();
         %current_dependencies_list = ();
         if (!($ext2 eq $file->{EXTENSION})) {
            if ($flat_layout) {
               print_header("$file->{FILENAME}.$ext2",":",$file->{NAMEyEXT});
            } else {
               print_header("$file->{PATHyNAME}.$ext2",":","$filename");
            }
            rec_print_dependencies(\%LISTOBJECT, $filename);
            #print STDOUT "\n";
            print_header("$file->{FILENAME}.$ext2",":","$file->{PATHyNAME}.$ext2") if ($short_target_names and $file->{FULLPATH_SRC} and !$flat_layout);
            print STDOUT "\n";
         }
      }
   }
}

#------------------------------------------------------------------------
# @rec_print_dependencies: 
# IN :
#   %0 = Hash of objects
#   $1 = Filename to print dependencies
# OUT: none
#------------------------------------------------------------------------
sub rec_print_dependencies {
   my $file = ${$_[0]}{$_[1]};
   for my $dep_filename (sort keys %{$file->{DEPENDENCIES}}) {
      my $dep_ptr = ${$file->{DEPENDENCIES}}{$dep_filename};
      my $tmp_filename = $dep_filename;
      $tmp_filename = "$dep_ptr->{PATHyNAME}.o" if ($dep_ptr->{COMPILABLE});
      my $tmp_filename0 = $tmp_filename;
      if ($flat_layout) {
         $tmp_filename0 = "$dep_ptr->{NAMEyEXT}";
         $tmp_filename0 = "$dep_ptr->{FILENAME}.o" if ($dep_ptr->{COMPILABLE});
      }
      #next if (($_[1] eq $dep_filename) or ($tmp_filename ~~ @current_dependencies_list));
      #next if (($_[1] eq $dep_filename) or mysmart($tmp_filename,\@current_dependencies_list));
      next if (($_[1] eq $dep_filename) or exists($current_dependencies_list{$tmp_filename}));

      print_item($tmp_filename0);
      #push @current_dependencies_list, $tmp_filename if (!mysmart($tmp_filename,\@current_dependencies_list));
      $current_dependencies_list{$tmp_filename} = 1 if !exists($current_dependencies_list{$tmp_filename});

      # Recursively call the function to print all depedencies
      rec_print_dependencies(\%{$_[0]}, $dep_filename) if (!$dep_ptr->{COMPILABLE});
   }
}

#------------------------------------------------------------------------
# @print_dep_rules_inv : reverse lookup
#------------------------------------------------------------------------
# sub print_dep_rules_inv {
# 	 #return 0;
#     print STDERR "Printing inverse dependencies\n" if ($msg >= 3);
#     for my $filename (keys %LISTOBJECT) {
#         my $file = $LISTOBJECT{$filename};
# 		  if ($flat_layout) {
# 		  		@dirdeplist{$file->{NAMEyEXT}} = ();
# 		  		@invdeplist{$file->{NAMEyEXT}} = ();
# 		  		# for my $depname (keys $file->{DEPENDENCIES}) {
# 		  		# 	 my $dep = $LISTOBJECT{$depname};
# 		  		# 	 push @{$dirdeplist{$file->{NAMEyEXT}}}, $dep->{NAMEyEXT};
# 		  		# }
# 				@current_dependencies_list = ();
# 				rec_fill_dirdeplist($filename, $filename);				
# 		  } else {
# 		  		print STDERR "\nERROR: Inv Dep for non flat layout is not yet implements\n";
# 		  		return 0;
# 		  }
# 	 }
#     for my $filename (keys %dirdeplist) {
# 		  for my $depname (@{$dirdeplist{$filename}}) {
# 				push @{$invdeplist{$depname}}, $filename;# if ($filename !~ @{$invdeplist{$depname}});
# 		  }
# 	 }
#     for my $depname (sort keys %invdeplist) {
# 		  if ($#{$invdeplist{$depname}} >= 0) {
# 				print_header("INVDEP_LIST_$depname","=","");
# 				for (@{$invdeplist{$depname}}) {
# 					 print_item($_);
# 				}
# 				print STDOUT "\n";
# 		  }
# 	 }
# }

# sub rec_fill_dirdeplist {
#     my $filename  = $_[0];
# 	 my $filename0 = $_[1];
# 	 my $file = $LISTOBJECT{$filename};
# 	 my $file0 = $LISTOBJECT{$filename0};
# 	 for my $depname (keys %file->{DEPENDENCIES}) {
# 		  my $dep = $LISTOBJECT{$depname};
# 		  next if (($depname eq $filename0) or ($depname ~~ @current_dependencies_list));

# 		  push @{$dirdeplist{$file0->{NAMEyEXT}}}, $dep->{NAMEyEXT};
# 		  push @current_dependencies_list, $filename;
#         # Recursively call the function to fill all depedencies
#         rec_fill_dirdeplist($depname, $filename0);
# 	 }
# }

sub print_dep_rules_inv2 {
   #return 0;
   print STDERR "Printing inverse dependencies\n" if ($msg >= 3);
   for my $filename (keys %LISTOBJECT) {
      my $file = $LISTOBJECT{$filename};
      if ($flat_layout) {
         @dirdeplist{$file->{NAMEyEXT}} = ();
         @invdeplist{$file->{NAMEyEXT}} = ();
         # for my $depname (keys $file->{DEPENDENCIES}) {
         # 	 my $dep = $LISTOBJECT{$depname};
         # 	 push @{$dirdeplist{$file->{NAMEyEXT}}}, $dep->{NAMEyEXT};
         # }
         #@current_dependencies_list = ();
         %current_dependencies_list = ();
         rec_fill_dirdeplist2($filename, $filename);				
      } else {
         print STDERR "\nWARNING: Inv Dep for non flat layout is not yet implements\n";
         return 0;
      }
   }
   for my $filename (keys %dirdeplist) {
      for my $depname (@{$dirdeplist{$filename}}) {
         push @{$invdeplist{$depname}}, $filename if (!mysmart($filename,\@{$invdeplist{$depname}}));
      }
   }
   for my $depname (sort keys %invdeplist) {
      #if ($#{$invdeplist{$depname}} >= 0) {
      my $mysize = scalar @{$invdeplist{$depname}};
      if ($mysize > 0) { 
      print STDOUT ".PHONY: _invdep_.".$depname."\n";
      print_header("_invdep_.".$depname,":","");
      for $fileyext (@{$invdeplist{$depname}}) {
         #if ($fileyext ~~  /(.*\/)*(.*)[.]([^.]*$)/) {
         if ($fileyext =~  /(.*\/)*(.*)[.]([^.]*$)/) {
            my $filn = ($2 ? $2 : "");
            my $exte = ($3 ? $3 : "");
            if ($SRCFile::TYPE_LIST{lc $exte} eq "COMPILABLE") {
               print_item($filn.".o");
            }
         }
      }
      print STDOUT "\n";
      }
   }
   for my $depname (sort keys %invdeplist) {
      #if ($#{$invdeplist{$depname}} >= 0) {
      my $mysize = scalar @{$invdeplist{$depname}};
      if ($mysize > 0) { 
         print_header("INVDEP_LIST_".$depname,"=","");
         for $fileyext (@{$invdeplist{$depname}}) {
            #if ($fileyext ~~  /(.*\/)*(.*)[.]([^.]*$)/) {
            if ($fileyext =~  /(.*\/)*(.*)[.]([^.]*$)/) {
               my $filn = ($2 ? $2 : "");
               my $exte = ($3 ? $3 : "");
               if ($SRCFile::TYPE_LIST{lc $exte} eq "COMPILABLE") {
                  print_item($filn.".o");
               }
            }
         }
         print STDOUT "\n";
      }
   }
}

sub rec_fill_dirdeplist2 {
   my $filename  = shift;
   my $filename0 = shift;
   my $file = $LISTOBJECT{$filename};
   my $file0 = $LISTOBJECT{$filename0};
   for my $depname (keys %{$file->{DEPENDENCIES}}) {
      my $dep = $LISTOBJECT{$depname};
      #next if (($depname eq $filename0) or ($depname ~~ @current_dependencies_list));
      #next if (($depname eq $filename0) or mysmart($depname,\@current_dependencies_list));
      next if (($depname eq $filename0) or exists($current_dependencies_list{$depname}));

      push @{$dirdeplist{$file0->{NAMEyEXT}}}, $dep->{NAMEyEXT} if (!mysmart($dep->{NAMEyEXT},\@{$dirdeplist{$file0->{NAMEyEXT}}}));
      #push @current_dependencies_list, $filename if (!mysmart($filename,\@current_dependencies_list));
      $current_dependencies_list{$filename} = 1 if !exists($current_dependencies_list{$depname});
      # Recursively call the function to fill all depedencies
      rec_fill_dirdeplist2($depname, $filename0);
   }
}

#------------------------------------------------------------------------
# @search_module: Find the key of the object that own the module name
# IN : $0 = Module name
# OUT: object key, undef otherwise.
#------------------------------------------------------------------------
sub search_module {
   my $module_name = shift;
   for (keys %LISTOBJECT) {
      return $_ if ($LISTOBJECT{$_}->has_module($module_name)); 
   }
   return undef;
}

#------------------------------------------------------------------------
# @search_unsolved_module: Find the key of the first object that has the module as one of his unsolved module list
# IN : $0 = Module name to be search
# OUT: object key, undef otherwise.
#------------------------------------------------------------------------
sub search_unsolved_module {
   my $module_name = shift;
   for (keys %LISTOBJECT) {
      return $_ if ($LISTOBJECT{$_}->SRCFile::has_unsolved_module($module_name));
   }
   return undef;
}

#------------------------------------------------------------------------
# @find_inc_file
# IN : 
#   $0 = file obj
#   $1 = supposed path to file to include
#   $2 = filename to include
# OUT: actual path to file; undef if not found
#------------------------------------------------------------------------
sub find_inc_file {
   my ($myfile, $mypath, $myfilename) = @_;
   for (@includelist) {
      if (-f $_.'/'.$mypath.$myfilename) {
         return $_.'/'.$mypath;
      } elsif (-f $_.'/'.$myfilename) {
         return $_.'/';
      }
   }
   if ($anywhere_inc) {
      for (keys %LISTOBJECT) {
         my $myobj=$LISTOBJECT{$_};
         if (($myfilename eq $myobj->{NAMEyEXT}) and 
             -f "$myobj->{PATHyNAMEyEXT}") {
            return $myobj->{FULLPATH_SRC};
         }
      }
   }
   if ($side_dir_inc) {
      my @mydirs = File::Spec->splitdir($myfile->{FULLPATH_SRC});
      for my $mysubdir ('*','*/*','*/*/*','*/*/*/*','*/*/*/*/*') {
         my @myfile2 = glob "$mydirs[0]/$mysubdir/$myfilename\n";
         if ($myfile2[0]) {
            # print STDERR "Found $myfilename in ".dirname($myfile2[0]) if ($msg >=4);
            return dirname($myfile2[0]).'/';
         }
      }
   }
   return undef;
}

#------------------------------------------------------------------------
# @print_missing: Print the missing module(s) / file(s) from the current tree
#------------------------------------------------------------------------
sub print_missing {
   #print STDERR "Includes missing from the current tree: ".join(" ",@outside_tree_list)."\n" if ($#outside_tree_list);
   my $missing_list_string = "";
   $missing_list_string = join(" ",@outside_tree_list) if ($#outside_tree_list);
   print STDERR "Includes missing from the current tree: ".$missing_list_string."\n" if ($missing_list_string);
   #TODO: do as module below, print first filename for each missing inc
   %module_missing = ();
   for my $filename (keys %LISTOBJECT) {
      my $file = $LISTOBJECT{$filename};
      for my $module (@{$file->{UNSOLVED_MODULE}}) {
         next if ($module eq "");
         next if (exists($module_missing_ignored{$module}));
         $module_missing{$module} = $file->{NAMEyEXT} if (!exists $module_missing{$module});
      }
   }
   if (keys %module_missing) {
      print STDERR "Modules missing from the current tree: ";
      while(my($module,$filename) = each(%module_missing)) {
         print STDERR "$module ($filename) ";
      }
      print STDERR "\n";
   }
}

#------------------------------------------------------------------------
# @print_unknown: Print Unknown module and use statement 
#------------------------------------------------------------------------
sub print_unknown {
   my $module_unknown = "";
   my $use_unknown = "";
   for my $filename (keys %LISTOBJECT) {
      my $file = $LISTOBJECT{$filename};
      while(my($line_number,$text_line) = each(%{$file->{UNKNOWN_MODULE}})) {
         $module_unknown .= "\t($filename) $line_number: $text_line\n";
      }
      while(my($line_number,$text_line) = each(%{$file->{UNKOWN_USE}})) {
         $use_unknown .= "\t($filename) $line_number: $text_line\n";
      }
   }
   print STDERR "Unknown module statement: \n".$module_unknown if ($module_unknown);
   print STDERR "Unknown use statement: \n".$use_unknown if ($use_unknown);
}

#------------------------------------------------------------------------
# @export_obj_list: Export a list of produced files (.o and .mod)
#------------------------------------------------------------------------
sub export_obj_list {
   open(my $EXPOUT,'>',$export_list);
   my @list_of_modules = ();
   for (keys %LISTOBJECT) {
      my $file = $LISTOBJECT{$_};
      if ($file->{COMPILABLE}) {
         if ($flat_layout) {
            print $EXPOUT "$file->{FILENAME}.o\n";
         } else {
            print $EXPOUT "$file->{PATHyNAME}.o\n";
         }
      }
      for (@{$file->{MODULE_LIST}}) {
         push @list_of_modules, $_ if $_ ne "" and (!mysmart($_,\@list_of_modules));
      }
   }
   for (sort @list_of_modules) {
      print $EXPOUT "$_.mod\n";
   }
   close($EXPOUT);
}

#########################################################################
# Main program beginning
#########################################################################

if ($output_file) {
   print STDERR "Redirecting STDOUT to $output_file\n" if ($msg>=3);
   open(STDOUT,">", "$output_file") or die "\nERROR: Can't redirect STDOUT\n";
}
@includelist = split(':',$include_dirs) if ($include_dirs);
push @includelist, $override_dir if $override_dir and (!mysmart($override_dir,\@includelist));

preproc_suppfile($suppress_errors_file);

preproc_srcfiles_overrides();
preproc_srcfiles();
my @objkeys = keys %LISTOBJECT;
my $cntall = 0;
my $cntprecent = 0;
my $cntprecent1 = 0;
my $cntstep = int(($#objkeys+1)/100)+1;
print STDERR "Process_files\n", if ($msg>=3);
while(my $filename = search_undone_file()) {
   if ($msg>=3) {
      $cntall++;
      $cntprecent1 = int(100.*$cntall/($#objkeys+1));
      if (int($cntprecent1/10) > int($cntprecent/10)) {
         $cntprecent = $cntprecent1;
         print STDERR $cntprecent,"% ";
         print STDERR "\n" if ($msg>=5);
      } elsif ($cntall % $cntstep == 0) {
         print STDERR "." if ($msg<5);
      }
   }
   process_file($filename);
}
print STDERR "\n" if ($msg>=3);
check_circular_dep();

print 'ifneq (,$(DEBUGMAKE))',"\n";
print '$(info ## ====================================================================)',"\n";
print '$(info ## File:',$output_file,")\n";
print '$(info ## )',"\n";
print 'endif',"\n";

print_files_list();
print_object_list();
print_module_list();
print_targets();
print_dep_rules();
print_dep_rules_inv2();

print_missing();
print_unknown();

print 'ifneq (,$(DEBUGMAKE))',"\n";
print '$(info ## ==== ',$output_file,' [END] =========================================)',"\n";
print 'endif',"\n";

export_obj_list() if ($export_list);

