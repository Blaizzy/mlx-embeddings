# Fish shell completions for Homebrew

# A note about aliases:
#
# * When defining completions for the (sub)commands themselves, only the full names are used, as they
#   are more descriptive and worth completing. Aliases are usually shorter than the full names, and
#   exist exactly to save time for users who already know what they want and are going to type the
#   command anyway (i.e. without completion).
# * Nevertheless, it's important to support aliases in the completions for their arguments/options.

##########################
## COMMAND LINE PARSING ##
##########################

function __fish_brew_args -d "Returns a list of all arguments given to brew"

    set -l tokens (commandline -opc)
    set -e tokens[1] # remove 'brew'
    for t in $tokens
        echo $t
    end
end

function __fish_brew_opts -d "Only arguments starting with a dash (options)"
    string match --all -- '-*' (__fish_brew_args)
end

# This can be used either to get the first argument or to match it against a given list of commands
#
# Usage examples (for `completion -n '...'`):
# * `__fish_brew_command` returns the command (first arg of brew) or exits with 1
# * `not __fish_brew_command` returns true when brew doesn't have a command yet
# * `__fish_brew_command list ls` returns true when brew command is _either_ `list` _or_ `ls`
#
function __fish_brew_command -d "Helps matching the first argument of brew"
    set args (__fish_brew_args)
    set -q args[1]; or return 1

    if count $argv
        contains -- $args[1] $argv
    else
        echo $args[1]
    end
end

function __fish_brew_subcommand -a cmd -d "Helps matching the second argument of brew"
    set args (__fish_brew_args)

    __fish_brew_command $cmd
    and set -q args[2]
    and set -l sub $args[2]
    or return 1

    set -e argv[1]
    if count $argv
        contains -- $sub $argv
    else
        echo $sub
    end
end

# This can be used to match any given option against the given list of arguments:
# * to add condition on interdependent options
# * to ddd condition on mutually exclusive options
#
# Usage examples (for `completion -n '...'`):
# * `__fish_brew_opt -s --long` returns true if _either_ `-s` _or_ `--long` is present
# * `not __fish_brew_opt --foo --bar` will work only if _neither_ `--foo` _nor_ `--bar` are present
#
function __fish_brew_opt -d "Helps matching brew options against the given list"

    not count $argv
    or contains -- $argv[1] (__fish_brew_opts)
    or begin
        set -q argv[2]
        and __fish_brew_opt $argv[2..-1]
    end
end


######################
## SUGGESTION LISTS ##
######################
# These functions return lists of suggestions for arguments completion

function __fish_brew_ruby_parse_json -a file parser -d 'Parses given JSON file with Ruby'
    # parser is any chain of methods to call on the parsed JSON
    ruby -e "require('json'); JSON.parse(File.read('$file'))$parser"
end

function __fish_brew_suggest_formulae_all -d 'Lists all available formulae with their descriptions'
    # store the brew cache path in a var (because calling (brew --cache) is slow)
    set -q __brew_cache_path
    or set -gx __brew_cache_path (brew --cache)

    # TODO: Probably drop this since I think that desc_cache.json is no longer generated. Is there a different available cache?
    if test -f "$__brew_cache_path/desc_cache.json"
        __fish_brew_ruby_parse_json "$__brew_cache_path/desc_cache.json" \
            '.each{ |k, v| puts([k, v].reject(&:nil?).join("\t")) }'
        # backup: (note that it lists only formulae names without descriptions)
    else
        brew search
    end
end

function __fish_brew_suggest_formulae_installed
    brew list
end

function __fish_brew_suggest_formulae_pinned
    brew list --pinned --versions \
        # replace first space with tab to make the following a description in the completions list:
        | string replace -r '\s' '\t'
end

function __fish_brew_suggest_formulae_unpinned
    # set difference of: all - pinned
    join -v2 (brew list --pinned | psub) (brew list | psub)
end

function __fish_brew_suggest_formulae_multiple_versions -d "List of installed formulae with their multiple versions"
    # NOTE: this assumes having `brew info --json=v1 --installed` cached
    # __fish_brew_ruby_parse_json 'installed.json' "
    #     .select{ |obj| obj['installed'].length > 1 }
    #     .each{ |obj| puts(
    #         obj['name'] +\"\t\"+
    #         obj['installed']
    #             .map{ |obj| obj['version'] }
    #             .join('; ')
    #     ) }
    # "

    # NOTE: this is bad because it's slower than calling `brew list --versions --multiple` and doesn't use any cache:
    # brew ruby -e 'Formula.installed.map{ |f| puts (f.full_name + "\t" + f.installed_kegs.map{ |keg| keg.version.to_s }.join(" ")) }'

    brew list --versions --multiple \
        # replace first space with tab to make the following a description in the completions list:
        | string replace -r '\s' '\t' \
        # a more visible versions separator:
        | string replace --all ' ' ', '
end

function __fish_brew_suggest_formula_versions -a formula -d "List of versions for a given formula"
    # NOTE: this assumes having `brew info --json=v1 --installed` cached
    # __fish_brew_ruby_parse_json 'installed.json' "
    #     .select{ |obj| obj['name'] == '$formula' }
    #     .each{ |obj| puts(obj['installed'].map{ |obj| obj['version'] }) }
    # "

    brew list --versions $formula \
        # cut off the first word in the output which is the formula name
        | string replace -r '\S+\s+' '' \
        # make it a list
        | string split ' '
end

function __fish_brew_suggest_formula_options -a formula -d "List installation options for a given formula"
    function list_pairs
        set -q argv[2]; or return 0
        echo $argv[1]\t$argv[2]
        set -e argv[1..2]
        list_pairs $argv
    end

    # brew options lists options name and its description on different lines
    list_pairs (brew options $formula | string trim)
end

function __fish_brew_suggest_formulae_outdated -d "List of outdated formulae with the information about potential upgrade"
    brew outdated --verbose \
        # replace first space with tab to make the following a description in the completions list:
        | string replace -r '\s' '\t'
end

function __fish_brew_suggest_taps_installed -d "List all available taps"
    brew tap
end

function __fish_brew_suggest_taps_pinned -d "List only pinned taps"
    brew tap --list-pinned
end

function __fish_brew_suggest_commands -d "Lists all commands names, including aliases"
    brew commands --quiet --include-aliases
end

# TODO: any better way to list available services?
function __fish_brew_suggest_services -d "Lists available services"
    set -l list (brew services list)
    set -e list[1] # Header
    for line in $list
        echo (string split ' ' $line)[1]
    end
end

function __fish_brew_suggest_casks_installed -d "Lists installed casks"
    brew cask list -1
end

function __fish_brew_suggest_casks_outdated -d "Lists outdated casks with the information about potential upgrade"
    brew cask outdated --verbose \
        # replace first space with tab to make the following a description in the completions list:
        | string replace -r '\s' '\t'
end

function __fish_brew_suggest_casks_all -d "Lists locally available casks"
    brew search --casks
end


##########################
## COMPLETION SHORTCUTS ##
##########################

function __fish_brew_complete_cmd -a cmd -d "A shortcut for defining brew commands completions"
    set -e argv[1]
    complete -f -c brew -n 'not __fish_brew_command' -a $cmd -d $argv
end

function __fish_brew_complete_arg -a cond -d "A shortcut for defining arguments completion for brew commands"
    set -e argv[1]
    # NOTE: $cond can be just a name of a command (or several) or additionally any other condition
    complete -f -c brew -n "__fish_brew_command $cond" $argv
end

function __fish_brew_complete_sub_cmd -a cmd sub -d "A shortcut for defining brew subcommands completions"
    set -e argv[1..2]
    __fish_brew_complete_arg "$cmd; and [ (count (__fish_brew_args)) = 1 ]" -a $sub -d $argv
end

function __fish_brew_complete_sub_arg -a cmd sub -d "A shortcut for defining brew subcommand arguments completions"
    set -e argv[1..2]
    # NOTE: $sub can be just a name of a subcommand (or several) or additionally any other condition
    complete -f -c brew -n "__fish_brew_subcommand $cmd $sub" $argv
end


##############
## COMMANDS ##
##############


__fish_brew_complete_cmd 'analytics' "User behaviour analytics commands"
__fish_brew_complete_sub_cmd 'analytics' 'state'           "Display analytics state"
__fish_brew_complete_sub_cmd 'analytics' 'on'              "Turn on analytics"
__fish_brew_complete_sub_cmd 'analytics' 'off'             "Turn off analytics"
__fish_brew_complete_sub_cmd 'analytics' 'regenerate-uuid' "Regenerate UUID used in analytics"


__fish_brew_complete_cmd 'cat' "Display the source to formula"
__fish_brew_complete_arg 'cat' -a '(__fish_brew_suggest_formulae_all)'


__fish_brew_complete_cmd 'cleanup' "Remove old installed versions"
__fish_brew_complete_arg 'cleanup' -a '(__fish_brew_suggest_formulae_installed)'
__fish_brew_complete_arg 'cleanup' -a '(__fish_brew_suggest_casks_installed)'
__fish_brew_complete_arg 'cleanup'      -l prune   -d "Remove all cache files older than given number of days" -a '(seq 1 5)'
__fish_brew_complete_arg 'cleanup' -s n -l dry-run -d "Show what files would be removed"
__fish_brew_complete_arg 'cleanup' -s s            -d "Scrub the cache, removing downloads for even the latest versions of formulae"


__fish_brew_complete_cmd 'command' "Display the path to command file"
__fish_brew_complete_arg 'command' -a '__fish_brew_suggest_commands'


__fish_brew_complete_cmd 'commands' "List built-in and external commands"
__fish_brew_complete_arg 'commands' -l quiet           -d "List only the names of commands without the header"
__fish_brew_complete_arg 'commands; and __fish_brew_opt --quiet' \
                                    -l include-aliases -d "The aliases of internal commands will be included"


__fish_brew_complete_cmd 'config' "Show Homebrew and system configuration for debugging"
# alias: --config


__fish_brew_complete_cmd 'deps' "Show dependencies for given formulae"
# accepts formulae argument only without --all or --installed options:
__fish_brew_complete_arg 'deps; and not __fish_brew_opt --all --installed' -a '(__fish_brew_suggest_formulae_all)'
# options that work only without --tree:
__fish_brew_complete_arg 'deps; and not __fish_brew_opt --tree' -s n         -d "Show in topological order"
__fish_brew_complete_arg 'deps; and not __fish_brew_opt --tree' -l 1         -d "Show only 1 level down"
__fish_brew_complete_arg 'deps; and not __fish_brew_opt --tree' -l union     -d "Show the union of dependencies for formulae, instead of the intersection"
__fish_brew_complete_arg 'deps; and not __fish_brew_opt --tree' -l full-name -d "List dependencies by their full name"
# --all and --installed are mutually exclusive:
__fish_brew_complete_arg 'deps; and not __fish_brew_opt --installed --tree' -l all       -d "Show dependencies for all formulae"
__fish_brew_complete_arg 'deps; and not __fish_brew_opt --all'              -l installed -d "Show dependencies for installed formulae"
# --tree works without options or with --installed
__fish_brew_complete_arg 'deps;
    and begin
        not __fish_brew_opts;
        or __fish_brew_opt --installed;
    end' -l tree -d "Show dependencies as tree"
# filters can be passed with any other options
__fish_brew_complete_arg 'deps' -l include-build    -d "Include the :build type dependencies"
__fish_brew_complete_arg 'deps' -l include-optional -d "Include the :optional type dependencies"
__fish_brew_complete_arg 'deps' -l skip-recommended -d "Skip :recommended  type  dependencies"


__fish_brew_complete_cmd 'desc' "Show formulae description or search by name and/or description"
__fish_brew_complete_arg 'desc; and [ (count (__fish_brew_args)) = 1 ]' -a '(__fish_brew_suggest_formulae_all)'
# FIXME: -n behaves differently from everything else
__fish_brew_complete_arg 'desc; and [ (count (__fish_brew_args)) = 1 ]' -s n -l name        -r -d "Search only names"
__fish_brew_complete_arg 'desc; and [ (count (__fish_brew_args)) = 1 ]' -s d -l description -r -d "Search only descriptions"
__fish_brew_complete_arg 'desc; and [ (count (__fish_brew_args)) = 1 ]' -s s -l search      -r -d "Search names and descriptions"


__fish_brew_complete_cmd 'diy' "Determine installation prefix for non-brew software"
__fish_brew_complete_arg 'diy configure' -l 'name=name'       -r -d "Set name of package"
__fish_brew_complete_arg 'diy configure' -l 'version=version' -r -d "Set version of package"


__fish_brew_complete_cmd 'doctor' "Check your system for potential problems"
# alias: dr


__fish_brew_complete_cmd 'fetch' "Download source packages for given formulae"
__fish_brew_complete_arg 'fetch' -a '(__fish_brew_suggest_formulae_all)'
__fish_brew_complete_arg 'fetch' -s f -l force        -d "Remove a previously cached version and re-fetch"
__fish_brew_complete_arg 'fetch' -l deps              -d "Also download dependencies"
__fish_brew_complete_arg 'fetch' -l build-from-source -d "Fetch source package instead of bottle"
__fish_brew_complete_arg 'fetch' -s v -l verbose      -d "Do a verbose VCS checkout"
__fish_brew_complete_arg 'fetch' -l retry             -d "Retry if a download fails or re-download if the checksum has changed"
# --HEAD and --devel are mutually exclusive:
__fish_brew_complete_arg 'fetch; and not __fish_brew_opt --devel --HEAD' -l devel -d "Download the development version from a VCS"
__fish_brew_complete_arg 'fetch; and not __fish_brew_opt --devel --HEAD' -l HEAD  -d "Download the HEAD version from a VCS"
# --build-from-source and --force-bottle are mutually exclusive:
__fish_brew_complete_arg 'fetch; and not __fish_brew_opt --force-bottle'    -s s -l build-from-source -d "Download the source rather than a bottle"
__fish_brew_complete_arg 'fetch; and not __fish_brew_opt --build-from-source -s' -l force-bottle      -d "Download a bottle if it exists"


__fish_brew_complete_cmd 'gist-logs' "Upload logs for a failed build of formula to a new Gist"
__fish_brew_complete_arg 'gist-logs' -a '(__fish_brew_suggest_formulae_all)'
__fish_brew_complete_arg 'gist-logs' -a '(__fish_brew_suggest_formulae_all)'
__fish_brew_complete_arg 'gist-logs' -s n -l new-issue -d "Also create a new issue in the appropriate GitHub repository"


__fish_brew_complete_cmd 'help' "Display help for given command"
__fish_brew_complete_arg 'help' -a '(__fish_brew_suggest_commands)'


__fish_brew_complete_cmd 'home' "Open Homebrew/formula's homepage"
__fish_brew_complete_arg 'home homepage' -a '(__fish_brew_suggest_formulae_all)'


__fish_brew_complete_cmd 'info' "Display information about formula"
# suggest formulae names only without --all/--installed options;
__fish_brew_complete_arg 'info abv; and not __fish_brew_opt --all --installed' -a '(__fish_brew_suggest_formulae_all)'
# --github or --json are applicable only without other options
__fish_brew_complete_arg 'info abv; and not __fish_brew_opts' -l github  -d "Open the GitHub History page for formula"
__fish_brew_complete_arg 'info abv; and not __fish_brew_opts' -l json=v1 -d "Print a JSON representation of formulae"
# --all and --installed require --json option and are mutually exclusive:
__fish_brew_complete_arg 'info abv;
    and begin
        __fish_brew_opt --json=v1;
        and not __fish_brew_opt --installed --all
    end' -l all       -d "Display JSON info for all formulae"
__fish_brew_complete_arg 'info abv;
    and begin
        __fish_brew_opt --json=v1;
        and not __fish_brew_opt --installed --all
    end' -l installed -d "Display JSON info for installed formulae"


__fish_brew_complete_cmd 'install' "Install formula"
# FIXME: install has a weird alias instal (with single l), probably it should also be supported
__fish_brew_complete_arg 'install' -a '(__fish_brew_suggest_formulae_all)'
# NOTE: upgrade command accepts same options as install
__fish_brew_complete_arg 'install upgrade' -s d -l debug -d "If install fails, open shell in temp directory"
# --env takes single obligatory argument:
__fish_brew_complete_arg 'install upgrade; and not __fish_brew_opt --env' -l env -r -d "Specify build environment" -a '
    std\t"Use standard build environment"
    super\t"Use superenv"
'
# --ignore-dependencies and --only-dependencies are mutually exclusive:
__fish_brew_complete_arg 'install upgrade;
    and not __fish_brew_opt --only-dependencies --ignore-dependencies
    ' -l ignore-dependencies -d "Skip installing any dependencies of any kind"
__fish_brew_complete_arg 'install upgrade;
    and not __fish_brew_opt --only-dependencies --ignore-dependencies
    ' -l only-dependencies   -d "Install dependencies but not the formula itself"
__fish_brew_complete_arg 'install upgrade' -l cc -d "Attempt to compile using the specified compiler" \
    -a 'clang gcc-4.0 gcc-4.2 gcc-4.3 gcc-4.4 gcc-4.5 gcc-4.6 gcc-4.7 gcc-4.8 gcc-4.9 llvm-gcc'
# --build-from-source and --force-bottle are mutually exclusive:
__fish_brew_complete_arg 'install upgrade; and not __fish_brew_opt --force-bottle'    -s s -l build-from-source -d "Compile the formula from source"
# FIXME: -s misbehaves allowing --force-bottle
__fish_brew_complete_arg 'install upgrade; and not __fish_brew_opt -s --build-from-source' -l force-bottle      -d "Install from a bottle if it exists"
# --HEAD and --devel are mutually exclusive:
__fish_brew_complete_arg 'install upgrade; and not __fish_brew_opt --devel --HEAD' -l devel -d "Install the development version"
__fish_brew_complete_arg 'install upgrade; and not __fish_brew_opt --devel --HEAD' -l HEAD  -d "Install the HEAD version"
__fish_brew_complete_arg 'install upgrade'      -l keep-tmp     -d "Keep temp files created during installation"
__fish_brew_complete_arg 'install upgrade'      -l build-bottle -d "Prepare the formula for eventual bottling during installation"
__fish_brew_complete_arg 'install upgrade' -s i -l interactive  -d "Download and patch formula, then open a shell"
__fish_brew_complete_arg 'install upgrade; and __fish_brew_opt -i --interactive' -s g -l git -d "Create a Git repository for working on patches"
# formula installation options are listed after the formula name:
__fish_brew_complete_arg 'install;
    and [ (count (__fish_brew_args)) -ge 2 ];
    and not string match --quiet -- "-*" (__fish_brew_args)[-1]
    ' -a '(__fish_brew_suggest_formula_options (__fish_brew_args)[-1])'


__fish_brew_complete_cmd 'irb' "Enter the interactive Homebrew Ruby shell"
__fish_brew_complete_arg 'irb' -l examples -d "Show several examples"


__fish_brew_complete_cmd 'leaves' "Installed formulae that are not dependencies of another installed formula"


__fish_brew_complete_cmd 'link' "Symlink installed formula files"
__fish_brew_complete_arg 'link ln' -a '(__fish_brew_suggest_formulae_installed)'
__fish_brew_complete_arg 'link ln'      -l overwrite -d "Overwrite existing files"
__fish_brew_complete_arg 'link ln' -s n -l dry-run   -d "Show what files would be linked or overwritten"
__fish_brew_complete_arg 'link ln' -s f -l force     -d "Allow keg-only formulae to be linked"


__fish_brew_complete_cmd 'linkapps' "Symlink .app bundles into /Applications (deprecated)"
__fish_brew_complete_arg 'linkapps' -a '(__fish_brew_suggest_formulae_installed)'
__fish_brew_complete_arg 'linkapps' -l local -d "Link into ~/Applications instead"


__fish_brew_complete_cmd 'list' "List installed formulae"
__fish_brew_complete_arg 'list ls' -a '(__fish_brew_suggest_formulae_installed)'
# --full-name or --unbrewed exclude any other arguments or options
__fish_brew_complete_arg 'list ls; and [ (count (__fish_brew_args)) = 1 ]' -l full-name -d "Print formulae with fully-qualified names"
__fish_brew_complete_arg 'list ls; and [ (count (__fish_brew_args)) = 1 ]' -l unbrewed  -d "List all files in the Homebrew prefix not installed by brew"
# --versions and --pinned work only with each other or alone
__fish_brew_complete_arg 'list ls;
    and begin
        not __fish_brew_opts;
        or      __fish_brew_opt --versions
        and not __fish_brew_opt --pinned
    end' -l pinned   -d "Show the versions of pinned formulae"
__fish_brew_complete_arg 'list ls;
    and begin
        not __fish_brew_opts;
        or      __fish_brew_opt --pinned
        and not __fish_brew_opt --versions
    end' -l versions -d "Show the version number"
# --multiple is an additional option for --versions
__fish_brew_complete_arg 'list ls;
    and     __fish_brew_opt --versions
    and not __fish_brew_opt --multiple
    ' -l multiple -d "Only show formulae with multiple versions"


__fish_brew_complete_cmd 'log' "Show git log for formula"
__fish_brew_complete_arg 'log' -a '(__fish_brew_suggest_formulae_all)'


__fish_brew_complete_cmd 'migrate' "Migrate renamed packages to new name"
# NOTE: should this work only with installed formulae?
__fish_brew_complete_arg 'migrate' -a '(__fish_brew_suggest_formulae_all)'
__fish_brew_complete_arg 'migrate' -s f -l force -d "Treat installed and passed formulae like if they are from same taps and migrate them anyway"


__fish_brew_complete_cmd 'missing' "Check given formula (or all) for missing dependencies"
__fish_brew_complete_arg 'missing' -a '(__fish_brew_suggest_formulae_installed)'
__fish_brew_complete_arg 'missing' -l hide -r -d "Act as if it's not installed" -a '(__fish_brew_suggest_formulae_installed)'


__fish_brew_complete_cmd 'options' "Display install options for formula"
__fish_brew_complete_arg 'options; and not __fish_brew_opt --installed --all' -a '(__fish_brew_suggest_formulae_all)'
__fish_brew_complete_arg 'options; and not __fish_brew_opt --installed --all' -l all       -d "Show options for all formulae"
__fish_brew_complete_arg 'options; and not __fish_brew_opt --installed --all' -l installed -d "Show options for all installed formulae"
__fish_brew_complete_arg 'options' -l compact -d "Show options as a space-delimited list"


__fish_brew_complete_cmd 'outdated' "Show formula that have updated version available"
__fish_brew_complete_arg 'outdated; and not __fish_brew_opt --quiet -v --verbose --json=v1'      -l quiet   -d "Display only names"
__fish_brew_complete_arg 'outdated; and not __fish_brew_opt --quiet -v --verbose --json=v1' -s v -l verbose -d "Display detailed version information"
__fish_brew_complete_arg 'outdated; and not __fish_brew_opt --quiet -v --verbose --json=v1'      -l json=v1 -d "Format output in JSON format"
# NOTE: check if this option requires a formula argument:
__fish_brew_complete_arg 'outdated' -l fetch-HEAD -d "Fetch the upstream repository to detect if the HEAD installation is outdated"


__fish_brew_complete_cmd 'pin' "Pin the specified formulae to their current versions"
__fish_brew_complete_arg 'pin' -a '(__fish_brew_suggest_formulae_unpinned)'


__fish_brew_complete_cmd 'postinstall' "Rerun the post-install steps for formula"
__fish_brew_complete_arg 'postinstall' -a '(__fish_brew_suggest_formulae_installed)'


__fish_brew_complete_cmd 'reinstall' "Uninstall and then install again"
__fish_brew_complete_arg 'reinstall' -a '(__fish_brew_suggest_formulae_installed)'


__fish_brew_complete_cmd 'search' "Display all locally available formulae or search by name/description"
__fish_brew_complete_arg 'search -S; and not __fish_brew_opts' -l desc -d "Search also in descriptions"
__fish_brew_complete_arg 'search -S; and not __fish_brew_opts' -l casks -d "Display all locally available casks"
for repo in debian fedora fink macports opensuse ubuntu
    __fish_brew_complete_arg "search -S; and not __fish_brew_opts" -l $repo -d "Search only in this repository"
end


__fish_brew_complete_cmd 'sh' "Start a Homebrew build environment shell"
__fish_brew_complete_arg 'sh' -l env=std -d "Use standard PATH instead of superenv's"


__fish_brew_complete_cmd 'style' "Check Homebrew style guidelines for formulae or files"
# NOTE: is it OK to use (ls) for suggestions?
__fish_brew_complete_arg 'style' -a '(ls)'                             -d "File"
__fish_brew_complete_arg 'style' -a '(__fish_brew_suggest_taps_installed)'     -d "Tap"
__fish_brew_complete_arg 'style' -a '(__fish_brew_suggest_formulae_installed)' -d "Formula"
__fish_brew_complete_arg 'style' -l fix -d "Use RuboCop's --auto-correct feature"
__fish_brew_complete_arg 'style' -l display-cop-names -d "Output RuboCop cop name for each violation"
# --only-cops and --except-cops are mutually exclusive:
__fish_brew_complete_arg 'style; and not __fish_brew_opt --only-cops --except-cops' -l only-cops   -d "Use only given Rubocop cops"
__fish_brew_complete_arg 'style; and not __fish_brew_opt --only-cops --except-cops' -l except-cops -d "Skip given Rubocop cops"


__fish_brew_complete_cmd 'switch' "Switch formula to another installed version"
# first argument is a formula with multiple versions:
__fish_brew_complete_arg 'switch; and [ (count (__fish_brew_args)) = 1 ]' -a '(__fish_brew_suggest_formulae_multiple_versions)'
# second argument is a list of versions for the previous argument:
__fish_brew_complete_arg 'switch; and [ (count (__fish_brew_args)) = 2 ]' -a '(__fish_brew_suggest_formula_versions (__fish_brew_args)[-1])'


__fish_brew_complete_cmd 'tap' "List installed taps or install a new tap"
__fish_brew_complete_arg 'tap; and not __fish_brew_opts' -l full          -d "Clone full repository instead of a shallow copy"
__fish_brew_complete_arg 'tap; and not __fish_brew_opts' -l repair        -d "Migrate tapped formulae from symlink-based to directory-based structure"
__fish_brew_complete_arg 'tap; and not __fish_brew_opts' -l list-official -d "List all official taps"
__fish_brew_complete_arg 'tap; and not __fish_brew_opts' -l list-pinned   -d "List all pinned taps"


__fish_brew_complete_cmd 'tap-info' "Display a brief summary of all installed taps"
__fish_brew_complete_arg 'tap-info; and not __fish_brew_opt --installed' -a '(__fish_brew_suggest_taps_installed)'
__fish_brew_complete_arg 'tap-info; and not __fish_brew_opt --installed' -l installed -d "Display information on all installed taps"
__fish_brew_complete_arg 'tap-info; and not __fish_brew_opt --json=v1'   -l json=v1   -d "Format output in JSON format"


__fish_brew_complete_cmd 'tap-pin' "Prioritize tap's formulae over core"
__fish_brew_complete_arg 'tap-pin' -a '(__fish_brew_suggest_taps_installed)'


__fish_brew_complete_cmd 'tap-unpin' "Don't prioritize tap's formulae over core anymore"
__fish_brew_complete_arg 'tap-unpin' -a '(__fish_brew_suggest_taps_pinned)'


__fish_brew_complete_cmd 'uninstall' "Uninstall formula"
# FIXME: uninstall has a weird alias uninstal (with single l), probably it should also be supported
__fish_brew_complete_arg 'uninstall remove rm' -a '(__fish_brew_suggest_formulae_installed)'
__fish_brew_complete_arg 'uninstall remove rm' -s f -l force               -d "Delete all installed versions"
__fish_brew_complete_arg 'uninstall remove rm'      -l ignore-dependencies -d "Won't fail, even if dependent formulae would still be installed"


__fish_brew_complete_cmd 'unlink' "Unlink formula"
__fish_brew_complete_arg 'unlink' -a '(__fish_brew_suggest_formulae_installed)'
__fish_brew_complete_arg 'unlink' -s n -l dry-run -d "Show what files would be unlinked"


__fish_brew_complete_cmd 'unlinkapps' "Remove symlinks created by brew linkapps (deprecated)"
__fish_brew_complete_arg 'unlinkapps' -a '(__fish_brew_suggest_formulae_installed)'
__fish_brew_complete_arg 'unlinkapps'      -l local   -d "Remove symlinks from ~/Applications"
__fish_brew_complete_arg 'unlinkapps' -s n -l dry-run -d "Show what symlinks would be removed"


__fish_brew_complete_cmd 'unpack' "Unpack formulae source files into current/given directory"
__fish_brew_complete_arg 'unpack' -a '(__fish_brew_suggest_formulae_all)'
__fish_brew_complete_arg 'unpack'      -l patch      -d "Apply patches to the unpacked source"
__fish_brew_complete_arg 'unpack' -s g -l git        -d "Initialize Git repository in the unpacked source"
__fish_brew_complete_arg 'unpack'      -l destdir -r -d "Unpack into the given directory" -a '(__fish_complete_directories "" "")'


__fish_brew_complete_cmd 'unpin' "Unpin formulae, allowing them to be upgraded"
__fish_brew_complete_arg 'unpin' -a '(__fish_brew_suggest_formulae_pinned)'


__fish_brew_complete_cmd 'untap' "Remove a tapped repository"
__fish_brew_complete_arg 'untap' -a '(__fish_brew_suggest_taps_installed)'


__fish_brew_complete_cmd 'update' "Fetch newest version of Homebrew and formulae"
__fish_brew_complete_arg 'update up'      -l merge -d "Use git merge (rather than git rebase)"
__fish_brew_complete_arg 'update up' -s f -l force -d "Always do a slower, full update check"


__fish_brew_complete_cmd 'upgrade' "Upgrade outdated brews"
__fish_brew_complete_arg 'upgrade' -a '(__fish_brew_suggest_formulae_outdated)'
__fish_brew_complete_arg 'upgrade' -l cleanup -d "Remove previously installed versions"
__fish_brew_complete_arg 'upgrade' -l fetch-HEAD -d "Fetch the upstream repository to detect if the HEAD installation is outdated"
# __fish_brew_complete_arg 'upgrade' -a '(complete -C"brew install -")'


__fish_brew_complete_cmd 'uses' "Show formulas that depend on specified formula"
__fish_brew_complete_arg 'uses' -a '(__fish_brew_suggest_formulae_all)'
__fish_brew_complete_arg 'uses' -l installed -d "List only installed formulae"
__fish_brew_complete_arg 'uses' -l recursive -d "Resolve more than one level of dependencies"
__fish_brew_complete_arg 'uses' -l include-build    -d "Include the :build type dependencies"
__fish_brew_complete_arg 'uses' -l include-optional -d "Include the :optional type dependencies"
__fish_brew_complete_arg 'uses' -l skip-recommended -d "Skip :recommended  type  dependencies"
# --HEAD and --devel are mutually exclusive:
__fish_brew_complete_arg 'uses; and not __fish_brew_opt --devel --HEAD' -l devel -d "Find cases development builds using formulae"
__fish_brew_complete_arg 'uses; and not __fish_brew_opt --devel --HEAD' -l HEAD  -d "Find cases HEAD builds using formulae"


__fish_brew_complete_cmd '--cache' "Display Homebrew/formula's cache location"
__fish_brew_complete_arg '--cache' -a '(__fish_brew_suggest_formulae_all)'


__fish_brew_complete_cmd '--cellar' "Display Homebrew/formula's Cellar path"
__fish_brew_complete_arg '--cellar' -a '(__fish_brew_suggest_formulae_all)'


__fish_brew_complete_cmd 'environment' "Summary of the Homebrew build environment"
# alias: --env
# NOTE: manpage lists --env and environment is just an alias, but I prefer to use full names in autocomplete


__fish_brew_complete_cmd '--prefix' "Display Homebrew/formula's install path"
__fish_brew_complete_arg '--prefix' -a '(__fish_brew_suggest_formulae_all)'


__fish_brew_complete_cmd '--repository' "Display Homebrew/tap's .git directory location"
__fish_brew_complete_arg '--repository --repo' -a '(__fish_brew_suggest_taps_installed)'


__fish_brew_complete_cmd '--version' "Display Homebrew's version number"


########################
## DEVELOPER COMMANDS ##
########################

__fish_brew_complete_cmd 'audit' "Check formulae for Homebrew coding style violations"
__fish_brew_complete_arg 'audit' -a '(__fish_brew_suggest_formulae_all)'
__fish_brew_complete_arg 'audit' -l strict            -d "Run additional checks (including RuboCop style checks)"
__fish_brew_complete_arg 'audit' -l fix               -d "Use RuboCop's --auto-correct feature"
__fish_brew_complete_arg 'audit' -l online            -d "Run additional checks that require a network connection"
__fish_brew_complete_arg 'audit' -l new-formula       -d "Check if a new formula is eligible for Homebrew"
__fish_brew_complete_arg 'audit' -l display-cop-names -d "Output RuboCop cop name for each violation"
__fish_brew_complete_arg 'audit' -l display-filename  -d "Prefix output lines with the file being audited"
# --only and --except are mutually exclusive:
# FIXME: not sure if these options can be repeated:
__fish_brew_complete_arg 'audit; and not __fish_brew_opt --only'   -l only   -d "Use only given audit method"
__fish_brew_complete_arg 'audit; and not __fish_brew_opt --except' -l except -d "Skip given audit method"
# --only-cops and --except-cops are mutually exclusive:
__fish_brew_complete_arg 'audit; and not __fish_brew_opt --only-cops --except-cops' -l only-cops   -d "Use only given Rubocop cops"
__fish_brew_complete_arg 'audit; and not __fish_brew_opt --only-cops --except-cops' -l except-cops -d "Skip given Rubocop cops"


__fish_brew_complete_cmd 'bottle' "Create a bottle (binary package)"
# FIXME: should it suggest all/installed formulae or only files with a cetain name?
__fish_brew_complete_arg 'bottle' -a '(__fish_brew_suggest_formulae_all)'
__fish_brew_complete_arg 'bottle; and not __fish_brew_opt --merge' -s v -l verbose -d "Print the bottling commands and any warnings encountered"
# --keep-old can be also used with --merged and is mutually exclusive with --no-rebuild
__fish_brew_complete_arg 'bottle; and not __fish_brew_opt --no-rebuild'       -l keep-old        -d "Keep rebuild version at its original value"
__fish_brew_complete_arg 'bottle; and not __fish_brew_opt --merge --keep-old' -l no-rebuild      -d "Remove rebuild version"
__fish_brew_complete_arg 'bottle; and not __fish_brew_opt --merge'            -l skip-relocation -d "Skip check if the bottle can be marked as relocatable"
__fish_brew_complete_arg 'bottle; and not __fish_brew_opt --merge'            -l root-url     -r -d "Specify the root of the bottle's URL instead of default"
__fish_brew_complete_arg 'bottle; and not __fish_brew_opt --merge'            -l force-core-tap  -d "Build a bottle even if formula is not in any installed taps"
# --merge is a separate mode of the bottle command:
__fish_brew_complete_arg 'bottle; and not __fish_brew_opt --merge' -l merge -d "Generate a bottle and print the new DSL merged into the existing formula"
__fish_brew_complete_arg 'bottle; and __fish_brew_opt --merge' -l write     -d "Write and commit the changes"
# --no-commit depends on --write (which depends on --merge):
__fish_brew_complete_arg 'bottle; and __fish_brew_opt --write' -l no-commit -d "Do not commit written changes"


__fish_brew_complete_cmd 'bump-formula-pr' "Create a pull request to update formula with a new URL or tag"
# FIXME: should it suggest all/installed formulae or only files with a cetain name?
__fish_brew_complete_arg 'bump-formula-pr' -a '(__fish_brew_suggest_formulae_all)'
__fish_brew_complete_arg 'bump-formula-pr'      -l devel   -d "Bump the development version instead of stable"
__fish_brew_complete_arg 'bump-formula-pr' -s n -l dry-run -d "Show what would be done"
# --write depends on --dry-run:
__fish_brew_complete_arg 'bump-formula-pr; and __fish_brew_opt -n --dry-run' -l write -d "Write changes but not commit them"
# --audit and --strict are mutually exclusive:
__fish_brew_complete_arg 'bump-formula-pr; and not __fish_brew_opt --audit --strict' -l audit  -d "Run audit before opening a PR"
__fish_brew_complete_arg 'bump-formula-pr; and not __fish_brew_opt --audit --strict' -l strict -d "Run audit --strict before opening a PR"
__fish_brew_complete_arg 'bump-formula-pr' -l mirror  -r -d "Specify mirror URL"
__fish_brew_complete_arg 'bump-formula-pr' -l version -r -d "Override the value parsed from the URL/tag"
__fish_brew_complete_arg 'bump-formula-pr' -l message -r -d "Append message to the default PR text"
# --url and --tag are mutually exclusive:
__fish_brew_complete_arg 'bump-formula-pr; and not __fish_brew_opt --url --tag --revision' -l url -r -d "Specify the URL"
__fish_brew_complete_arg 'bump-formula-pr; and not __fish_brew_opt --url --tag --sha-256'  -l tag -r -d "Specify the tag"
# --sha-256 and --revision depend on --url and --tag correspondingly:
__fish_brew_complete_arg 'bump-formula-pr; and __fish_brew_opt --url' -l sha-256  -r -d "Specify checksum of the new download"
__fish_brew_complete_arg 'bump-formula-pr; and __fish_brew_opt --tag' -l revision -r -d "Specify revision corresponding to the tag"


__fish_brew_complete_cmd 'create' "Create new formula from URL and open it in the editor"
# all options have to be passed after the URL argument:
# --autotools --cmake and --meson are mutually exclusive:
__fish_brew_complete_arg 'create; and [ (count (__fish_brew_args)) -ge 2 ];
    and not __fish_brew_opt --autotools --cmake --meson'           -l autotools      -d "Use template for Autotools-style build"
__fish_brew_complete_arg 'create; and [ (count (__fish_brew_args)) -ge 2 ];
    and not __fish_brew_opt --autotools --cmake --meson'           -l cmake          -d "Use template for CMake-style build"
__fish_brew_complete_arg 'create; and [ (count (__fish_brew_args)) -ge 2 ];
    and not __fish_brew_opt --autotools --cmake --meson'           -l meson          -d "Use template for Meson-style build"
__fish_brew_complete_arg 'create; and [ (count (__fish_brew_args)) -ge 2 ]' -l no-fetch       -d "Don't download URL to the cache"
__fish_brew_complete_arg 'create; and [ (count (__fish_brew_args)) -ge 2 ]' -l set-name    -r -d "Set name explicitly"
__fish_brew_complete_arg 'create; and [ (count (__fish_brew_args)) -ge 2 ]' -l set-version -r -d "Set version explicitly"
__fish_brew_complete_arg 'create; and [ (count (__fish_brew_args)) -ge 2 ]' -l tap         -r -d "Specify tap for the generated formula"


__fish_brew_complete_cmd 'edit' "Open Homebrew/formula for editing"
__fish_brew_complete_arg 'edit' -a '(__fish_brew_suggest_formulae_all)'


__fish_brew_complete_cmd 'formula' "Display the path where formula is located"
__fish_brew_complete_arg 'formula' -a '(__fish_brew_suggest_formulae_all)'


__fish_brew_complete_cmd 'linkage' "Check library links of an installed formula"
__fish_brew_complete_arg 'linkage' -a '(__fish_brew_suggest_formulae_installed)'
__fish_brew_complete_arg 'linkage' -l test    -d "Only display missing libraries"
__fish_brew_complete_arg 'linkage' -l reverse -d "Print the dylib followed by the binaries which link to it"


__fish_brew_complete_cmd 'man' "Generate Homebrew's manpages"
__fish_brew_complete_arg 'man' -l fail-if-changed -d "Fail if changes are detected in the manpage outputs"


# TODO: it could use Ruby to autocomplete Github PRs from homebrew/core (patch-source arg)
__fish_brew_complete_cmd 'pull' "Apply a patch from GitHub to Homebrew"
__fish_brew_complete_arg 'pull' -l bottle                  -d "Pull the bottle-update commit and publish files on Bintray"
__fish_brew_complete_arg 'pull' -l bump                    -d "For one-formula PRs, reword commit message to the preferred format"
__fish_brew_complete_arg 'pull' -l clean                   -d "Don't rewrite or modify commits in the pulled PR"
__fish_brew_complete_arg 'pull' -l ignore-whitespace       -d "Silently ignore whitespace discrepancies when applying diffs"
__fish_brew_complete_arg 'pull' -l resolve                 -d "Manually resolve patch application failures (instead of aborting)"
__fish_brew_complete_arg 'pull' -l branch-okay             -d "Don't warn if pulling to a branch besides master"
__fish_brew_complete_arg 'pull' -l no-pbcopy               -d "Don't copy anything to the system clipboard"
__fish_brew_complete_arg 'pull' -l no-publish              -d "Don't publish bottles to Bintray"
__fish_brew_complete_arg 'pull' -l warn-on-publish-failure -d "Don't exit if there's a failure publishing bottles on Bintray"


__fish_brew_complete_cmd 'release-notes' "List merged PRs on Homebrew/brew between two Git refs"
__fish_brew_complete_arg 'release-notes' -l markdown -d "Output as a Markdown list"
# TODO: suggest Git tags as arguments


__fish_brew_complete_cmd 'tap-new' "Generate template files for a new tap"


__fish_brew_complete_cmd 'test' "Run tests for given formula"
__fish_brew_complete_arg 'test' -a '(__fish_brew_suggest_formulae_installed)'
__fish_brew_complete_arg 'test' -s d -l debug    -d "Test with an interative debugger"
__fish_brew_complete_arg 'test'      -l keep-tmp -d "Don't delete temp files created for the test"
# --HEAD and --devel are mutually exclusive:
__fish_brew_complete_arg 'test; and not __fish_brew_opt --devel --HEAD' -l devel -d "Test the development version"
__fish_brew_complete_arg 'test; and not __fish_brew_opt --devel --HEAD' -l HEAD  -d "Test the HEAD version"


__fish_brew_complete_cmd 'tests' "Run Homebrew's unit and integration tests"
__fish_brew_complete_arg 'tests' -s v -l verbose      -d "Print the command that runs the tests"
__fish_brew_complete_arg 'tests' -l coverage          -d "Also generate code coverage reports"
__fish_brew_complete_arg 'tests' -l generic           -d "Only run OS-agnostic tests"
__fish_brew_complete_arg 'tests' -l no-compat         -d "Don't load the compatibility layer"
__fish_brew_complete_arg 'tests' -l only           -r -d "Run only specified *_spec.rb"
__fish_brew_complete_arg 'tests' -l seed           -r -d "Randomize tests with the given seed"
__fish_brew_complete_arg 'tests' -l online            -d "Include tests that use the GitHub API"
__fish_brew_complete_arg 'tests' -l official-cmd-taps -d "Include tests that use any of the taps for official external commands"


__fish_brew_complete_cmd 'update-test' "Run a test of brew update with a new repository clone"
__fish_brew_complete_arg 'update-test' -l commit -r -d "Specify start commit (instead of default origin/master)"
__fish_brew_complete_arg 'update-test' -l before -r -d "Specify date of the start commit"
__fish_brew_complete_arg 'update-test' -l to-tag    -d "Set HOMEBREW_UPDATE_TO_TAG to test updating between tags"
__fish_brew_complete_arg 'update-test' -l keep-tmp  -d "Keep the temp directory with the new repository clone"


#########################
## ADDITIONAL COMMANDS ##
#########################
# These commands are not in the manpage, but bew lists them in brew commands
# NOTE: I'm not even sure if these commands should be listed


# FIXME: I don't have aspell installed, is it a part of the core homebrew?
__fish_brew_complete_cmd 'aspell-dictionaries' "Generate new dictionaries for the aspell formula"


__fish_brew_complete_cmd 'mirror' "Reupload stable URL for a formula to Bintray to use as a mirror"
# FIXME: should it suggest all/installed formulae or only files with a cetain name?
__fish_brew_complete_arg 'mirror' -a '(__fish_brew_suggest_formulae_all)'
# TODO: find description for the test option
__fish_brew_complete_arg 'mirror' -l test # -d ???


__fish_brew_complete_cmd 'readall' "Import all formulae in core/given tap"
__fish_brew_complete_arg 'readall' -a '(__fish_brew_suggest_taps_installed)'


# NOTE: update-report: The Ruby implementation of brew update. Never called manually.


__fish_brew_complete_cmd 'update-reset' "Fetches and resets Homebrew and all taps to their latest origin/master"


__fish_brew_complete_cmd 'vendor-install' "Install vendor version of Homebrew dependencies"


################################
## OFFICIAL EXTERNAL COMMANDS ##
################################
# TODO: These commands are installed/tapped separately, so they should be completed only when present

##############
### BUNDLE ###

__fish_brew_complete_cmd 'bundle' "Install or upgrade all dependencies in a Brewfile"
__fish_brew_complete_arg 'bundle; and [ (count (__fish_brew_args)) = 1 ]' -s v -l verbose -d "Print more details"

# --file/--global option is available for bundle command and all its subcommands except exec
__fish_brew_complete_arg 'bundle;
        and not __fish_brew_subcommand bundle exec;
        and not __fish_brew_opt --file --global
    ' -l file -r -d "Specify Brewfile"
__fish_brew_complete_arg 'bundle;
        and not __fish_brew_subcommand bundle exec;
        and not __fish_brew_opt --file --global
    ' -l global  -d "Use \$HOME/.Brewfile"

__fish_brew_complete_sub_cmd 'bundle' 'dump'    "Write all installed casks/formulae/taps into a Brewfile"
__fish_brew_complete_sub_cmd 'bundle' 'cleanup' "Uninstall all dependencies not listed in a Brewfile"
__fish_brew_complete_sub_cmd 'bundle' 'check'   "Check if all dependencies are installed in a Brewfile"
__fish_brew_complete_sub_cmd 'bundle' 'exec'    "Run an external command in an isolated build environment"

# --force is available only for the dump/cleanup subcommands
__fish_brew_complete_sub_arg 'bundle' 'dump cleanup' -l force -d "Uninstall dependencies or overwrite an existing Brewfile"

# --no-upgrade is available for bundle command and its check subcommand
__fish_brew_complete_arg 'bundle; and [ (count (__fish_brew_args)) = 1 ];
        or __fish_brew_subcommand bundle check
    ' -l no-upgrade -d "Don't run brew upgrade for outdated dependencies"


############
### CASK ###

__fish_brew_complete_cmd 'cask' "Install macOS applications distributed as binaries"

__fish_brew_complete_sub_cmd 'cask' '--version' "Display the Homebrew-Cask version"

__fish_brew_complete_sub_cmd 'cask' 'audit'     "Verify installability of Casks"

__fish_brew_complete_sub_cmd 'cask' 'cat'       "Dump raw source of the given Cask to the standard output"

__fish_brew_complete_sub_cmd 'cask' 'create'    "Create the given Cask and open it in an editor"

__fish_brew_complete_sub_cmd 'cask' 'doctor'    "Check for configuration issues"

__fish_brew_complete_sub_cmd 'cask' 'edit'      "Edit the given Cask"

__fish_brew_complete_sub_cmd 'cask' 'fetch'     "Download remote application files to local cache"
__fish_brew_complete_sub_arg 'cask' 'fetch' -l force -d "Redownload even if the files are already cached"

__fish_brew_complete_sub_cmd 'cask' 'home'      "Open the homepage of the given Cask"

__fish_brew_complete_sub_cmd 'cask' 'info'      "Display information about the given Cask"

__fish_brew_complete_sub_cmd 'cask' 'install'   "Install the given Cask"
__fish_brew_complete_sub_arg 'cask' 'install' -l force          -d "Reinstall even if the Cask is already present"
__fish_brew_complete_sub_arg 'cask' 'install' -l skip-cask-deps -d "Skip any Cask dependencies"
__fish_brew_complete_sub_arg 'cask' 'install' -l require-sha    -d "Abort if the Cask doesn't define a checksum"
__fish_brew_complete_sub_arg 'cask' 'audit install' -rl language -d "Set language of the Cask to install. The first matching language is used, otherwise the default language on the Cask. The default value is the language of your system."

__fish_brew_complete_sub_cmd 'cask' 'list'      "List installed Casks or staged files of the given installed Casks"
__fish_brew_complete_sub_arg 'cask' 'list ls' -s 1        -d "Format output in a single column"
__fish_brew_complete_sub_arg 'cask' 'list ls' -l versions -d "Show all installed versions"

__fish_brew_complete_sub_cmd 'cask' 'outdated'  "List the outdated installed Casks"
__fish_brew_complete_sub_arg 'cask' 'outdated upgrade' -l greedy -d "Include the Casks having auto_updates true or version :latest"
__fish_brew_complete_sub_arg 'cask' 'outdated; and not __fish_brew_opt --verbose --quiet' -l verbose -d "Display outdated and the latest version"
__fish_brew_complete_sub_arg 'cask' 'outdated; and not __fish_brew_opt --verbose --quiet' -l quiet   -d "Suppress versions from the output"

__fish_brew_complete_sub_cmd 'cask' 'reinstall' "Reinstall the given Cask"

__fish_brew_complete_sub_cmd 'cask' 'style'     "Check Cask style using RuboCop"
__fish_brew_complete_sub_arg 'cask' 'style' -l fix -d "Auto-correct any style errors if possible"

__fish_brew_complete_sub_cmd 'cask' 'upgrade'     "Upgrades all outdated casks"
__fish_brew_complete_sub_arg 'cask' 'upgrade' -l force
__fish_brew_complete_sub_arg 'cask' 'upgrade' -l dry-run

__fish_brew_complete_sub_cmd 'cask' 'uninstall' "Uninstall the given Cask"
__fish_brew_complete_sub_arg 'cask' 'uninstall remove rm' -l force -d "Uninstall even if the Cask is not present"

__fish_brew_complete_sub_cmd 'cask' 'zap'       "Zap all files associated with the given Cask"

# Common argument for these commands: either all available, only installed cask tokens, or outdated casks:
__fish_brew_complete_sub_arg 'cask' 'audit cat edit fetch home homepage info abv install style' -a '(__fish_brew_suggest_casks_all)'
__fish_brew_complete_sub_arg 'cask' 'list ls reinstall outdated uninstall remove rm zap'        -a '(__fish_brew_suggest_casks_installed)'
__fish_brew_complete_sub_arg 'cask' 'upgrade'        -a '(__fish_brew_suggest_casks_outdated)'


################
### SERVICES ###

__fish_brew_complete_cmd 'services' "Integrates Homebrew formulae with macOS's launchctl manager"
__fish_brew_complete_arg 'services; and [ (count (__fish_brew_args)) = 1 ]' -s v -l verbose -d "Print more details"

__fish_brew_complete_sub_cmd 'services' 'list'    "List all running services for the current user"
__fish_brew_complete_sub_cmd 'services' 'run'     "Run service without starting at login/boot"
__fish_brew_complete_sub_cmd 'services' 'start'   "Start service immediately and register it to launch at login/boot"
__fish_brew_complete_sub_cmd 'services' 'stop'    "Stop service immediately and unregister it from launching at login/boot"
__fish_brew_complete_sub_cmd 'services' 'restart' "Stop and start service immediately and register it to launch at login/boot"
__fish_brew_complete_sub_cmd 'services' 'cleanup' "Remove all unused services"

__fish_brew_complete_sub_arg 'services' 'run start stop restart' -l all -d "Run all available services"
__fish_brew_complete_sub_arg 'services' 'run start stop restart' -a '(__fish_brew_suggest_services)'
