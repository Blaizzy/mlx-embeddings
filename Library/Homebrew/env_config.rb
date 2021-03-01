# typed: true
# frozen_string_literal: true

module Homebrew
  # Helper module for querying Homebrew-specific environment variables.
  #
  # @api private
  module EnvConfig
    extend T::Sig

    module_function

    ENVS = {
      HOMEBREW_ARCH:                          {
        description: "Linux only: Pass this value to a type name representing the compiler's `-march` option.",
        default:     "native",
      },
      HOMEBREW_ARTIFACT_DOMAIN:               {
        description: "Prefix all download URLs, including those for bottles, with this value. " \
                     "For example, `HOMEBREW_ARTIFACT_DOMAIN=http://localhost:8080` will cause a " \
                     "formula with the URL `https://example.com/foo.tar.gz` to instead download from " \
                     "`http://localhost:8080/example.com/foo.tar.gz`.",
      },
      HOMEBREW_AUTO_UPDATE_SECS:              {
        description: "Automatically check for updates once per this seconds interval.",
        default:     300,
      },
      HOMEBREW_BAT:                           {
        description: "If set, use `bat` for the `brew cat` command.",
        boolean:     true,
      },
      HOMEBREW_BAT_CONFIG_PATH:               {
        description:  "Use this as the `bat` configuration file.",
        default_text: "`$HOME/.bat/config`.",
      },
      HOMEBREW_BINTRAY_KEY:                   {
        description: "Use this API key when accessing the Bintray API (where bottles are stored).",
      },
      HOMEBREW_BINTRAY_USER:                  {
        description: "Use this username when accessing the Bintray API (where bottles are stored).",
      },
      HOMEBREW_BOOTSNAP:                      {
        description: "If set, use Bootsnap to speed up repeated `brew` calls. "\
                     "A no-op when using Homebrew's vendored, relocatable Ruby on macOS (as it doesn't work).",
        boolean:     true,
      },
      HOMEBREW_BOTTLE_DOMAIN:                 {
        description:  "Use this URL as the download mirror for bottles. " \
                      "If bottles at that URL are temporarily unavailable, " \
                      "the default bottle domain will be used as a fallback mirror. " \
                      "For example, `HOMEBREW_BOTTLE_DOMAIN=http://localhost:8080` will cause all bottles to " \
                      "download from the prefix `http://localhost:8080/`. " \
                      "If bottles are not available at `HOMEBREW_BOTTLE_DOMAIN` " \
                      "they will be downloaded from the default bottle domain.",
        default_text: "macOS: `https://homebrew.bintray.com/`, " \
                      "Linux: `https://linuxbrew.bintray.com/`.",
        default:      HOMEBREW_BOTTLE_DEFAULT_DOMAIN,
      },
      HOMEBREW_BREW_GIT_REMOTE:               {
        description: "Use this URL as the Homebrew/brew `git`(1) remote.",
        default:     HOMEBREW_BREW_DEFAULT_GIT_REMOTE,
      },
      HOMEBREW_BROWSER:                       {
        description:  "Use this as the browser when opening project homepages.",
        default_text: "`$BROWSER` or the OS's default browser.",
      },
      HOMEBREW_CACHE:                         {
        description:  "Use this directory as the download cache.",
        default_text: "macOS: `$HOME/Library/Caches/Homebrew`, " \
                      "Linux: `$XDG_CACHE_HOME/Homebrew` or `$HOME/.cache/Homebrew`.",
        default:      HOMEBREW_DEFAULT_CACHE,
      },
      HOMEBREW_CASK_OPTS:                     {
        description: "Append these options to all `cask` commands. All `--*dir` options, " \
                     "`--language`, `--require-sha`, `--no-quarantine` and `--no-binaries` are supported. " \
                     "For example, you might add something like the following to your " \
                     "`~/.profile`, `~/.bash_profile`, or `~/.zshenv`:\n\n" \
                     '    `export HOMEBREW_CASK_OPTS="--appdir=~/Applications --fontdir=/Library/Fonts"`',
      },
      HOMEBREW_CLEANUP_PERIODIC_FULL_DAYS:    {
        description: "If set, `brew install`, `brew upgrade` and `brew reinstall` will cleanup all formulae " \
                     "when this number of days has passed.",
        default:     30,
      },
      HOMEBREW_CLEANUP_MAX_AGE_DAYS:          {
        description: "Cleanup all cached files older than this many days.",
        default:     120,
      },
      HOMEBREW_COLOR:                         {
        description: "If set, force colour output on non-TTY outputs.",
        boolean:     true,
      },
      HOMEBREW_CORE_GIT_REMOTE:               {
        description:  "Use this URL as the Homebrew/homebrew-core `git`(1) remote.",
        default_text: "macOS: `https://github.com/Homebrew/homebrew-core`, " \
                      "Linux: `https://github.com/Homebrew/linuxbrew-core`.",
        default:      HOMEBREW_CORE_DEFAULT_GIT_REMOTE,
      },
      HOMEBREW_CURLRC:                        {
        description: "If set, do not pass `--disable` when invoking `curl`(1), which disables the " \
                     "use of `curlrc`.",
        boolean:     true,
      },
      HOMEBREW_CURL_RETRIES:                  {
        description: "Pass the given retry count to `--retry` when invoking `curl`(1).",
        default:     3,
      },
      HOMEBREW_CURL_VERBOSE:                  {
        description: "If set, pass `--verbose` when invoking `curl`(1).",
        boolean:     true,
      },
      HOMEBREW_DEVELOPER:                     {
        description: "If set, tweak behaviour to be more relevant for Homebrew developers (active or " \
                     "budding) by e.g. turning warnings into errors.",
        boolean:     true,
      },
      HOMEBREW_DISABLE_LOAD_FORMULA:          {
        description: "If set, refuse to load formulae. This is useful when formulae are not trusted (such " \
                     "as in pull requests).",
        boolean:     true,
      },
      HOMEBREW_DISPLAY:                       {
        description:  "Use this X11 display when opening a page in a browser, for example with " \
                      "`brew home`. Primarily useful on Linux.",
        default_text: "`$DISPLAY`.",
      },
      HOMEBREW_DISPLAY_INSTALL_TIMES:         {
        description: "If set, print install times for each formula at the end of the run.",
        boolean:     true,
      },
      HOMEBREW_EDITOR:                        {
        description:  "Use this editor when editing a single formula, or several formulae in the " \
                      "same directory." \
                      "\n\n    *Note:* `brew edit` will open all of Homebrew as discontinuous files " \
                      "and directories. Visual Studio Code can handle this correctly in project mode, but many " \
                      "editors will do strange things in this case.",
        default_text: "`$EDITOR` or `$VISUAL`.",
      },
      HOMEBREW_FAIL_LOG_LINES:                {
        description: "Output this many lines of output on formula `system` failures.",
        default:     15,
      },
      HOMEBREW_FORBIDDEN_LICENSES:            {
        description: "A space-separated list of licenses. Homebrew will refuse to install a " \
                     "formula if it or any of its dependencies has a license on this list.",
      },
      HOMEBREW_FORCE_BREWED_CURL:             {
        description: "If set, always use a Homebrew-installed `curl`(1) rather than the system version. " \
                     "Automatically set if the system version of `curl` is too old.",
        boolean:     true,
      },
      HOMEBREW_FORCE_BREWED_GIT:              {
        description: "If set, always use a Homebrew-installed `git`(1) rather than the system version. " \
                     "Automatically set if the system version of `git` is too old.",
        boolean:     true,
      },
      HOMEBREW_FORCE_HOMEBREW_ON_LINUX:       {
        description: "If set, running Homebrew on Linux will use URLs for macOS. This is useful when merging " \
                     "pull requests for macOS while on Linux.",
        boolean:     true,
      },
      HOMEBREW_FORCE_VENDOR_RUBY:             {
        description: "If set, always use Homebrew's vendored, relocatable Ruby version even if the system version " \
                     "of Ruby is new enough.",
        boolean:     true,
      },
      HOMEBREW_GITHUB_API_TOKEN:              {
        description: "Use this personal access token for the GitHub API, for features such as " \
                     "`brew search`. You can create one at <https://github.com/settings/tokens>. If set, " \
                     "GitHub will allow you a greater number of API requests. For more information, see: " \
                     "<https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting>" \
                     "\n\n    *Note:* Homebrew doesn't require permissions for any of the scopes, but some " \
                     "developer commands may require additional permissions.",
      },
      HOMEBREW_GIT_EMAIL:                     {
        description: "Set the Git author and committer email to this value.",
      },
      HOMEBREW_GIT_NAME:                      {
        description: "Set the Git author and committer name to this value.",
      },
      HOMEBREW_INSTALL_BADGE:                 {
        description:  "Print this text before the installation summary of each successful build.",
        default_text: 'The "Beer Mug" emoji.',
        default:      "🍺",
      },
      HOMEBREW_INTERNET_ARCHIVE_KEY:          {
        description: "Use this API key when accessing the Internet Archive S3 API, where bottles are stored. " \
                     "The format is access:secret. See https://archive.org/account/s3.php",
      },
      HOMEBREW_LIVECHECK_WATCHLIST:           {
        description: "Consult this file for the list of formulae to check by default when no formula argument " \
                     "is passed to `brew livecheck`.",
        default:     "$HOME/.brew_livecheck_watchlist",
      },
      HOMEBREW_LOGS:                          {
        description:  "Use this directory to store log files.",
        default_text: "macOS: `$HOME/Library/Logs/Homebrew`, " \
                      "Linux: `$XDG_CACHE_HOME/Homebrew/Logs` or `$HOME/.cache/Homebrew/Logs`.",
        default:      HOMEBREW_DEFAULT_LOGS,
      },
      HOMEBREW_MAKE_JOBS:                     {
        description:  "Use this value as the number of parallel jobs to run when building with `make`(1).",
        default_text: "The number of available CPU cores.",
        default:      lambda {
          require "os"
          require "hardware"
          Hardware::CPU.cores
        },
      },
      HOMEBREW_NO_ANALYTICS:                  {
        description: "If set, do not send analytics. For more information, see: <https://docs.brew.sh/Analytics>",
        boolean:     true,
      },
      HOMEBREW_NO_AUTO_UPDATE:                {
        description: "If set, do not automatically update before running some commands e.g. " \
                     "`brew install`, `brew upgrade` and `brew tap`.",
        boolean:     true,
      },
      HOMEBREW_NO_BOOTSNAP:                   {
        description: "If set, do not use Bootsnap to speed up repeated `brew` calls.",
        boolean:     true,
      },
      HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK: {
        description: "If set, do not check for broken dependents after installing, upgrading or reinstalling " \
                     "formulae.",
        boolean:     true,
      },
      HOMEBREW_NO_COLOR:                      {
        description:  "If set, do not print text with colour added.",
        default_text: "`$NO_COLOR`.",
        boolean:      true,
      },
      HOMEBREW_NO_COMPAT:                     {
        description: "If set, disable all use of legacy compatibility code.",
        boolean:     true,
      },
      HOMEBREW_NO_EMOJI:                      {
        description: "If set, do not print `HOMEBREW_INSTALL_BADGE` on a successful build." \
                     "\n\n    *Note:* Will only try to print emoji on OS X Lion or newer.",
        boolean:     true,
      },
      HOMEBREW_NO_GITHUB_API:                 {
        description: "If set, do not use the GitHub API, e.g. for searches or fetching relevant issues " \
                     "after a failed install.",
        boolean:     true,
      },
      HOMEBREW_NO_INSECURE_REDIRECT:          {
        description: "If set, forbid redirects from secure HTTPS to insecure HTTP." \
                     "\n\n    *Note:* While ensuring your downloads are fully secure, this is likely to cause " \
                     "from-source SourceForge, some GNU & GNOME-hosted formulae to fail to download.",
        boolean:     true,
      },
      HOMEBREW_NO_INSTALL_CLEANUP:            {
        description: "If set, `brew install`, `brew upgrade` and `brew reinstall` will never automatically " \
                     "cleanup installed/upgraded/reinstalled formulae or all formulae every " \
                     "`HOMEBREW_CLEANUP_PERIODIC_FULL_DAYS` days.",
        boolean:     true,
      },
      HOMEBREW_PRY:                           {
        description: "If set, use Pry for the `brew irb` command.",
        boolean:     true,
      },
      HOMEBREW_SKIP_OR_LATER_BOTTLES:         {
        description: "If set along with `HOMEBREW_DEVELOPER`, do not use bottles from older versions " \
                     "of macOS. This is useful in development on new macOS versions.",
        boolean:     true,
      },
      HOMEBREW_SORBET_RUNTIME:                {
        description: "If set, enable runtime typechecking using Sorbet.",
        boolean:     true,
      },
      HOMEBREW_SVN:                           {
        description:  "Use this as the `svn`(1) binary.",
        default_text: "A Homebrew-built Subversion (if installed), or the system-provided binary.",
      },
      HOMEBREW_TEMP:                          {
        description:  "Use this path as the temporary directory for building packages. Changing " \
                      "this may be needed if your system temporary directory and Homebrew prefix are on " \
                      "different volumes, as macOS has trouble moving symlinks across volumes when the target " \
                      "does not yet exist. This issue typically occurs when using FileVault or custom SSD " \
                      "configurations.",
        default_text: "macOS: `/private/tmp`, Linux: `/tmp`.",
        default:      HOMEBREW_DEFAULT_TEMP,
      },
      HOMEBREW_UPDATE_REPORT_ONLY_INSTALLED:  {
        description: "If set, `brew update` only lists updates to installed software.",
        boolean:     true,
      },
      HOMEBREW_UPDATE_TO_TAG:                 {
        description: "If set, always use the latest stable tag (even if developer commands " \
                     "have been run).",
        boolean:     true,
      },
      HOMEBREW_VERBOSE:                       {
        description: "If set, always assume `--verbose` when running commands.",
        boolean:     true,
      },
      HOMEBREW_DEBUG:                         {
        description: "If set, always assume `--debug` when running commands.",
        boolean:     true,
      },
      HOMEBREW_VERBOSE_USING_DOTS:            {
        description: "If set, verbose output will print a `.` no more than once a minute. This can be " \
                     "useful to avoid long-running Homebrew commands being killed due to no output.",
        boolean:     true,
      },
      all_proxy:                              {
        description: "Use this SOCKS5 proxy for `curl`(1), `git`(1) and `svn`(1) when downloading through Homebrew.",
      },
      ftp_proxy:                              {
        description: "Use this FTP proxy for `curl`(1), `git`(1) and `svn`(1) when downloading through Homebrew.",
      },
      http_proxy:                             {
        description: "Use this HTTP proxy for `curl`(1), `git`(1) and `svn`(1) when downloading through Homebrew.",
      },
      https_proxy:                            {
        description: "Use this HTTPS proxy for `curl`(1), `git`(1) and `svn`(1) when downloading through Homebrew.",
      },
      no_proxy:                               {
        description: "A comma-separated list of hostnames and domain names excluded " \
                     "from proxying by `curl`(1), `git`(1) and `svn`(1) when downloading through Homebrew.",
      },
      SUDO_ASKPASS:                           {
        description: "If set, pass the `-A` option when calling `sudo`(8).",
      },
    }.freeze

    def env_method_name(env, hash)
      method_name = env.to_s
                       .sub(/^HOMEBREW_/, "")
                       .downcase
      method_name = "#{method_name}?" if hash[:boolean]
      method_name
    end

    CUSTOM_IMPLEMENTATIONS = %w[
      HOMEBREW_MAKE_JOBS
      HOMEBREW_CASK_OPTS
    ].freeze

    ENVS.each do |env, hash|
      method_name = env_method_name(env, hash)
      env = env.to_s

      if hash[:boolean]
        define_method(method_name) do
          ENV[env].present?
        end
      elsif hash[:default].present?
        # Needs a custom implementation.
        next if CUSTOM_IMPLEMENTATIONS.include?(env)

        define_method(method_name) do
          ENV[env].presence || hash.fetch(:default).to_s
        end
      else
        define_method(method_name) do
          ENV[env].presence
        end
      end
    end

    # Needs a custom implementation.
    sig { returns(String) }
    def make_jobs
      jobs = ENV["HOMEBREW_MAKE_JOBS"].to_i
      return jobs.to_s if jobs.positive?

      ENVS.fetch(:HOMEBREW_MAKE_JOBS)
          .fetch(:default)
          .call
          .to_s
    end

    sig { returns(T::Array[String]) }
    def cask_opts
      Shellwords.shellsplit(ENV.fetch("HOMEBREW_CASK_OPTS", ""))
    end

    sig { returns(T::Boolean) }
    def cask_opts_binaries?
      cask_opts.reverse_each do |opt|
        return true if opt == "--binaries"
        return false if opt == "--no-binaries"
      end

      true
    end

    sig { returns(T::Boolean) }
    def cask_opts_quarantine?
      cask_opts.reverse_each do |opt|
        return true if opt == "--quarantine"
        return false if opt == "--no-quarantine"
      end

      true
    end

    sig { returns(T::Boolean) }
    def cask_opts_require_sha?
      cask_opts.include?("--require-sha")
    end
  end
end
