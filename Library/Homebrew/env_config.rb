# frozen_string_literal: true

module Homebrew
  module EnvConfig
    module_function

    ENVS = {
      HOMEBREW_ARCH:                      {
        description: "Linux only: Homebrew will pass the set value to type name to the compiler's `-march` option.",
        default:     "native",
      },
      HOMEBREW_ARTIFACT_DOMAIN:           {
        description: "Instructs Homebrew to prefix all download URLs, including those for bottles, with this " \
                     "variable. For example, `HOMEBREW_ARTIFACT_DOMAIN=http://localhost:8080` will cause a " \
                     "formula with the URL `https://example.com/foo.tar.gz` to instead download from " \
                     "`http://localhost:8080/example.com/foo.tar.gz`.",
      },
      HOMEBREW_AUTO_UPDATE_SECS:          {
        description: "Homebrew will only check for autoupdates once per this seconds interval.",
        default:     300,
      },
      HOMEBREW_BAT:                       {
        description: "Homebrew will use `bat` for the `brew cat` command.",
        boolean:     true,
      },
      HOMEBREW_BINTRAY_KEY:               {
        description: "Homebrew uses this API key when accessing the Bintray API (where bottles are stored).",
      },
      HOMEBREW_BINTRAY_USER:              {
        description: "Homebrew uses this username when accessing the Bintray API (where bottles are stored).",
      },
      HOMEBREW_BOTTLE_DOMAIN:             {
        description:  "Instructs Homebrew to use the specified URL as its download mirror for bottles. " \
        "For example, `HOMEBREW_BOTTLE_DOMAIN=http://localhost:8080` will cause all bottles to " \
        "download from the prefix `http://localhost:8080/`.",
        default_text: "macOS: `https://homebrew.bintray.com/`, Linux: `https://linuxbrew.bintray.com/`.",
        default:      HOMEBREW_BOTTLE_DEFAULT_DOMAIN,
      },
      HOMEBREW_BREW_GIT_REMOTE:           {
        description: "Instructs Homebrew to use the specified URL as its Homebrew/brew `git`(1) remote.",
        default:     HOMEBREW_BREW_DEFAULT_GIT_REMOTE,
      },
      HOMEBREW_BROWSER:                   {
        description:  "Homebrew uses this setting as the browser when opening project homepages.",
        default_text: "`$BROWSER` or the OS's default browser.",
      },
      HOMEBREW_CACHE:                     {
        description:  "Instructs Homebrew to use the specified directory as the download cache.",
        default_text: "macOS: `$HOME/Library/Caches/Homebrew`, " \
                      "Linux: `$XDG_CACHE_HOME/Homebrew` or `$HOME/.cache/Homebrew`.",
        default:      HOMEBREW_DEFAULT_CACHE,
      },
      HOMEBREW_COLOR:                     {
        description: "Homebrew force colour output on non-TTY outputs.",
        boolean:     true,
      },
      HOMEBREW_CORE_GIT_REMOTE:           {
        description:  "instructs Homebrew to use the specified URL as its Homebrew/homebrew-core `git`(1) remote.",
        default_text: "macOS: `https://github.com/Homebrew/homebrew-core`, " \
                      "Linux: `https://github.com/Homebrew/linuxbrew-core`.",
        default:      HOMEBREW_CORE_DEFAULT_GIT_REMOTE,
      },
      HOMEBREW_CURLRC:                    {
        description: "Homebrew will not pass `--disable` when invoking `curl`(1), which disables the " \
                     "use of `curlrc`.",
        boolean:     true,
      },
      HOMEBREW_CURL_RETRIES:              {
        description: "Homebrew will pass the given retry count to `--retry` when invoking `curl`(1).",
        default:     3,
      },
      HOMEBREW_CURL_VERBOSE:              {
        description: "Homebrew will pass `--verbose` when invoking `curl`(1).",
        boolean:     true,
      },
      HOMEBREW_DEVELOPER:                 {
        description: "Homebrew will tweak behaviour to be more relevant for Homebrew developers (active or " \
                     "budding), e.g. turning warnings into errors.",
        boolean:     true,
      },
      HOMEBREW_DISABLE_LOAD_FORMULA:      {
        description: "Homebrew will refuse to load formulae. This is useful when formulae are not trusted (such " \
                     "as in pull requests).",
        boolean:     true,
      },
      HOMEBREW_DISPLAY:                   {
        description:  "Homebrew will use this X11 display when opening a page in a browser, for example with " \
                     "`brew home`. Primarily useful on Linux.",
        default_text: "`$DISPLAY`.",
      },
      HOMEBREW_DISPLAY_INSTALL_TIMES:     {
        description: "Homebrew will print install times for each formula at the end of the run.",
        boolean:     true,
      },
      HOMEBREW_EDITOR:                    {
        description:  "Homebrew will use this editor when editing a single formula, or several formulae in the " \
                     "same directory.\n\n    *Note:* `brew edit` will open all of Homebrew as discontinuous files " \
                     "and directories. Visual Studio Code can handle this correctly in project mode, but many " \
                     "editors will do strange things in this case.",
        default_text: "`$EDITOR` or `$VISUAL`.",
      },
      HOMEBREW_FAIL_LOG_LINES:            {
        description: "Homebrew will output this many lines of output on formula `system` failures.",
        default:     15,
      },
      HOMEBREW_FORCE_BREWED_CURL:         {
        description: "Homebrew will always use a Homebrew-installed `curl`(1) rather than the system version. " \
                     "Automatically set if the system version of `curl` is too old.",
      },
      HOMEBREW_FORCE_BREWED_GIT:          {
        description: "Homebrew will always use a Homebrew-installed `git`(1) rather than the system version. " \
                     "Automatically set if the system version of `git` is too old.",
      },
      HOMEBREW_FORCE_HOMEBREW_ON_LINUX:   {
        description: "Homebrew running on Linux will use URLs for Homebrew on macOS. This is useful when merging" \
                     "pull requests on Linux for macOS.",
        boolean:     true,
      },
      HOMEBREW_FORCE_VENDOR_RUBY:         {
        description: "Homebrew will always use its vendored, relocatable Ruby version even if the system version " \
                     "of Ruby is new enough.",
        boolean:     true,
      },
      HOMEBREW_GITHUB_API_PASSWORD:       {
        description: "GitHub password for authentication with the GitHub API, used by Homebrew for features" \
                     "such as `brew search`. We strongly recommend using `HOMEBREW_GITHUB_API_TOKEN` instead.",
      },
      HOMEBREW_GITHUB_API_TOKEN:          {
        description: "A personal access token for the GitHub API, used by Homebrew for features such as " \
                     "`brew search`. You can create one at <https://github.com/settings/tokens>. If set, " \
                     "GitHub will allow you a greater number of API requests. For more information, see: " \
                     "<https://developer.github.com/v3/#rate-limiting>\n\n    *Note:* Homebrew doesn't " \
                     "require permissions for any of the scopes.",
      },
      HOMEBREW_GITHUB_API_USERNAME:       {
        description: "GitHub username for authentication with the GitHub API, used by Homebrew for features " \
                     "such as `brew search`. We strongly recommend using `HOMEBREW_GITHUB_API_TOKEN` instead.",
      },
      HOMEBREW_GIT_EMAIL:                 {
        description: "Homebrew will set the Git author and committer name to this value.",
      },
      HOMEBREW_GIT_NAME:                  {
        description: "Homebrew will set the Git author and committer email to this value.",
      },
      HOMEBREW_INSTALL_BADGE:             {
        description:  "Text printed before the installation summary of each successful build.",
        default_text: 'The "Beer Mug" emoji.',
        default:      "🍺",
      },
      HOMEBREW_LOGS:                      {
        description:  "IHomebrew will use the specified directory to store log files.",
        default_text: "macOS: `$HOME/Library/Logs/Homebrew`, "\
                      "Linux: `$XDG_CACHE_HOME/Homebrew/Logs` or `$HOME/.cache/Homebrew/Logs`.",
        default:      HOMEBREW_DEFAULT_LOGS,
      },
      HOMEBREW_MAKE_JOBS:                 {
        description:  "Instructs Homebrew to use the value of `HOMEBREW_MAKE_JOBS` as the number of " \
                      "parallel jobs to run when building with `make`(1).",
        default_text: "The number of available CPU cores.",
        default:      lambda {
          require "os"
          require "hardware"
          Hardware::CPU.cores
        },
      },
      HOMEBREW_NO_ANALYTICS:              {
        description: "Homebrew will not send analytics. See: <https://docs.brew.sh/Analytics>.",
        boolean:     true,
      },
      HOMEBREW_NO_AUTO_UPDATE:            {
        description: "Homebrew will not auto-update before running `brew install`, `brew upgrade` or `brew tap`.",
        boolean:     true,
      },
      HOMEBREW_NO_BOTTLE_SOURCE_FALLBACK: {
        description: "Homebrew will fail on the failure of installation from a bottle rather than " \
                     "falling back to building from source.",
        boolean:     true,
      },
      HOMEBREW_NO_COLOR:                  {
        description:  "Homebrew will not print text with colour added.",
        default_text: "`$NO_COLOR`.",
        boolean:      true,
      },
      HOMEBREW_NO_COMPAT:                 {
        description: "Homebrew disables all use of legacy compatibility code.",
        boolean:     true,
      },
      HOMEBREW_NO_EMOJI:                  {
        description: "Homebrew will not print the `HOMEBREW_INSTALL_BADGE` on a successful build." \
                     "\n\n    *Note:* Homebrew will only try to print emoji on OS X Lion or newer.",
        boolean:     true,
      },
      HOMEBREW_NO_GITHUB_API:             {
        description: "Homebrew will not use the GitHub API, e.g. for searches or fetching relevant issues " \
                     "on a failed install.",
        boolean:     true,
      },
      HOMEBREW_NO_INSECURE_REDIRECT:      {
        description: "Homebrew will not permit redirects from secure HTTPS to insecure HTTP." \
                     "\n\n    *Note:* While ensuring your downloads are fully secure, this is likely to cause " \
                     "from-source SourceForge, some GNU & GNOME based formulae to fail to download.",
        boolean:     true,
      },
      HOMEBREW_NO_INSTALL_CLEANUP:        {
        description: "`brew install`, `brew upgrade` and `brew reinstall` will never automatically cleanup " \
                     "installed/upgraded/reinstalled formulae or all formulae every 30 days.",
        boolean:     true,
      },
      HOMEBREW_PRY:                       {
        description: "Homebrew will use Pry for the `brew irb` command.",
        boolean:     true,
      },
      HOMEBREW_SKIP_OR_LATER_BOTTLES:     {
        description: "Along with `HOMEBREW_DEVELOPER` Homebrew will not use bottles from older versions of macOS. " \
                     "This is useful in Homebrew development on new macOS versions.",
        boolean:     true,
      },
      HOMEBREW_SVN:                       {
        description: "Forces Homebrew to use a particular `svn` binary. Otherwise, a Homebrew-built Subversion " \
                     "if installed, or the system-provided binary.",
      },
      HOMEBREW_TEMP:                      {
        description:  "Instructs Homebrew to use `HOMEBREW_TEMP` as the temporary directory for building " \
                      "packages. This may be needed if your system temp directory and Homebrew prefix are on " \
                      "different volumes, as macOS has trouble moving symlinks across volumes when the target" \
                      "does not yet exist. This issue typically occurs when using FileVault or custom SSD" \
                      "configurations.",
        default_text: "macOS: `/private/tmp`, Linux: `/tmp`.",
        default:      HOMEBREW_DEFAULT_TEMP,
      },
      HOMEBREW_UPDATE_TO_TAG:             {
        description: "Instructs Homebrew to always use the latest stable tag (even if developer commands " \
                     "have been run).",
        boolean:     true,
      },
      HOMEBREW_VERBOSE:                   {
        description: "Homebrew always assumes `--verbose` when running commands.",
        boolean:     true,
      },
      HOMEBREW_VERBOSE_USING_DOTS:        {
        boolean:     true,
        description: "Homebrew's verbose output will print a `.` no more than once a minute. This can be " \
                     "useful to avoid long-running Homebrew commands being killed due to no output.",
      },
      all_proxy:                          {
        description: "Sets the SOCKS5 proxy to be used by `curl`(1), `git`(1) and `svn`(1) when downloading " \
                     "through Homebrew.",
      },
      ftp_proxy:                          {
        description: "Sets the FTP proxy to be used by `curl`(1), `git`(1) and `svn`(1) when downloading " \
                     "through Homebrew.",
      },
      http_proxy:                         {
        description: "Sets the HTTP proxy to be used by `curl`(1), `git`(1) and `svn`(1) when downloading " \
                     "through Homebrew.",
      },
      https_proxy:                        {
        description: "Sets the HTTPS proxy to be used by `curl`(1), `git`(1) and `svn`(1) when downloading " \
                     "through Homebrew.",
      },
      no_proxy:                           {
        description: "Sets the comma-separated list of hostnames and domain names that should be excluded " \
                     "from proxying by `curl`(1), `git`(1) and `svn`(1) when downloading through Homebrew.",
      },
    }.freeze

    def env_method_name(env, hash)
      method_name = env.to_s
                       .sub(/^HOMEBREW_/, "")
                       .downcase
      method_name = "#{method_name}?" if hash[:boolean]
      method_name
    end

    ENVS.each do |env, hash|
      method_name = env_method_name(env, hash)
      env = env.to_s

      if hash[:boolean]
        define_method(method_name) do
          ENV[env].present?
        end
      elsif hash[:default].present?
        # Needs a custom implementation.
        next if env == "HOMEBREW_MAKE_JOBS"

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
    def make_jobs
      jobs = ENV["HOMEBREW_MAKE_JOBS"].to_i
      return jobs.to_s if jobs.positive?

      ENVS.fetch(:HOMEBREW_MAKE_JOBS)
          .fetch(:default)
          .call
          .to_s
    end
  end
end
