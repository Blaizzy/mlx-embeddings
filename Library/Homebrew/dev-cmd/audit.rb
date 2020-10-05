# frozen_string_literal: true

require "formula"
require "formula_versions"
require "utils/curl"
require "utils/github/actions"
require "utils/shared_audits"
require "utils/spdx"
require "extend/ENV"
require "formula_cellar_checks"
require "cmd/search"
require "style"
require "date"
require "missing_formula"
require "digest"
require "cli/parser"
require "json"

module Homebrew
  module_function

  def audit_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `audit` [<options>] [<formula>]

        Check <formula> for Homebrew coding style violations. This should be run before
        submitting a new formula. If no <formula> are provided, check all locally
        available formulae and skip style checks. Will exit with a non-zero status if any
        errors are found.
      EOS
      switch "--strict",
             description: "Run additional, stricter style checks."
      switch "--git",
             description: "Run additional, slower style checks that navigate the Git repository."
      switch "--online",
             description: "Run additional, slower style checks that require a network connection."
      switch "--new-formula",
             description: "Run various additional style checks to determine if a new formula is eligible "\
                          "for Homebrew. This should be used when creating new formula and implies "\
                          "`--strict` and `--online`."
      flag   "--tap=",
             description: "Check the formulae within the given tap, specified as <user>`/`<repo>."
      switch "--fix",
             description: "Fix style violations automatically using RuboCop's auto-correct feature."
      switch "--display-cop-names",
             description: "Include the RuboCop cop name for each violation in the output."
      switch "--display-filename",
             description: "Prefix every line of output with the file or formula name being audited, to "\
                          "make output easy to grep."
      switch "--skip-style",
             description: "Skip running non-RuboCop style checks. Useful if you plan on running "\
                          "`brew style` separately. Default unless a formula is specified by name"
      switch "-D", "--audit-debug",
             description: "Enable debugging and profiling of audit methods."
      comma_array "--only",
                  description: "Specify a comma-separated <method> list to only run the methods named "\
                               "`audit_`<method>."
      comma_array "--except",
                  description: "Specify a comma-separated <method> list to skip running the methods named "\
                               "`audit_`<method>."
      comma_array "--only-cops",
                  description: "Specify a comma-separated <cops> list to check for violations of only the listed "\
                               "RuboCop cops."
      comma_array "--except-cops",
                  description: "Specify a comma-separated <cops> list to skip checking for violations of the listed "\
                               "RuboCop cops."

      conflicts "--only", "--except"
      conflicts "--only-cops", "--except-cops", "--strict"
      conflicts "--only-cops", "--except-cops", "--only"
      conflicts "--display-cop-names", "--skip-style"
      conflicts "--display-cop-names", "--only-cops"
      conflicts "--display-cop-names", "--except-cops"
    end
  end

  def audit
    args = audit_args.parse

    Homebrew.auditing = true
    inject_dump_stats!(FormulaAuditor, /^audit_/) if args.audit_debug?

    formula_count = 0
    problem_count = 0
    corrected_problem_count = 0
    new_formula_problem_count = 0
    new_formula = args.new_formula?
    strict = new_formula || args.strict?
    online = new_formula || args.online?
    git = args.git?
    skip_style = args.skip_style? || args.no_named? || args.tap

    ENV.activate_extensions!
    ENV.setup_build_environment

    audit_formulae = if args.tap
      Tap.fetch(args.tap).formula_names.map { |name| Formula[name] }
    elsif args.no_named?
      Formula
    else
      args.named.to_resolved_formulae
    end
    style_files = args.named.to_formulae_paths unless skip_style

    only_cops = args.only_cops
    except_cops = args.except_cops
    options = { fix: args.fix?, debug: args.debug?, verbose: args.verbose? }

    if only_cops
      options[:only_cops] = only_cops
    elsif args.new_formula?
      nil
    elsif except_cops
      options[:except_cops] = except_cops
    elsif !strict
      options[:except_cops] = [:FormulaAuditStrict]
    end

    # Check style in a single batch run up front for performance
    style_offenses = Style.check_style_json(style_files, options) if style_files
    # load licenses
    spdx_license_data = SPDX.license_data
    spdx_exception_data = SPDX.exception_data
    new_formula_problem_lines = []
    audit_formulae.sort.each do |f|
      only = only_cops ? ["style"] : args.only
      options = {
        new_formula:         new_formula,
        strict:              strict,
        online:              online,
        git:                 git,
        only:                only,
        except:              args.except,
        spdx_license_data:   spdx_license_data,
        spdx_exception_data: spdx_exception_data,
      }
      options[:style_offenses] = style_offenses.for_path(f.path) if style_offenses
      options[:display_cop_names] = args.display_cop_names?
      options[:build_stable] = args.build_stable?

      fa = FormulaAuditor.new(f, options)
      fa.audit
      next if fa.problems.empty? && fa.new_formula_problems.empty?

      formula_count += 1
      problem_count += fa.problems.size
      problem_lines = format_problem_lines(fa.problems)
      corrected_problem_count = options[:style_offenses].count(&:corrected?) if options[:style_offenses]
      new_formula_problem_lines = format_problem_lines(fa.new_formula_problems)
      if args.display_filename?
        puts problem_lines.map { |s| "#{f.path}: #{s}" }
      else
        puts "#{f.full_name}:", problem_lines.map { |s| "  #{s}" }
      end

      next unless ENV["GITHUB_ACTIONS"]

      (fa.problems + fa.new_formula_problems).each do |message:, location:|
        annotation = GitHub::Actions::Annotation.new(
          :error, message, file: f.path, line: location&.line, column: location&.column
        )
        puts annotation if annotation.relevant?
      end
    end

    new_formula_problem_count += new_formula_problem_lines.size
    puts new_formula_problem_lines.map { |s| "  #{s}" }

    total_problems_count = problem_count + new_formula_problem_count
    problem_plural = "#{total_problems_count} #{"problem".pluralize(total_problems_count)}"
    formula_plural = "#{formula_count} #{"formula".pluralize(formula_count)}"
    corrected_problem_plural = "#{corrected_problem_count} #{"problem".pluralize(corrected_problem_count)}"
    errors_summary = "#{problem_plural} in #{formula_plural} detected"
    errors_summary += ", #{corrected_problem_plural} corrected" if corrected_problem_count.positive?

    ofail errors_summary if problem_count.positive? || new_formula_problem_count.positive?
  end

  def format_problem_lines(problems)
    problems.uniq
            .map { |message:, location:| format_problem(message, location) }
  end

  def format_problem(message, location)
    "* #{location&.to_s&.dup&.concat(": ")}#{message.chomp.gsub("\n", "\n    ")}"
  end

  class FormulaText
    def initialize(path)
      @text = path.open("rb", &:read)
      @lines = @text.lines.to_a
    end

    def without_patch
      @text.split("\n__END__").first
    end

    def trailing_newline?
      /\Z\n/ =~ @text
    end

    def =~(other)
      other =~ @text
    end

    def include?(s)
      @text.include? s
    end

    def line_number(regex, skip = 0)
      index = @lines.drop(skip).index { |line| line =~ regex }
      index ? index + 1 : nil
    end

    def reverse_line_number(regex)
      index = @lines.reverse.index { |line| line =~ regex }
      index ? @lines.count - index : nil
    end
  end

  class FormulaAuditor
    include FormulaCellarChecks

    attr_reader :formula, :text, :problems, :new_formula_problems

    def initialize(formula, options = {})
      @formula = formula
      @versioned_formula = formula.versioned_formula?
      @new_formula_inclusive = options[:new_formula]
      @new_formula = options[:new_formula] && !@versioned_formula
      @strict = options[:strict]
      @online = options[:online]
      @build_stable = options[:build_stable]
      @git = options[:git]
      @display_cop_names = options[:display_cop_names]
      @only = options[:only]
      @except = options[:except]
      # Accept precomputed style offense results, for efficiency
      @style_offenses = options[:style_offenses]
      # Allow the formula tap to be set as homebrew/core, for testing purposes
      @core_tap = formula.tap&.core_tap? || options[:core_tap]
      @problems = []
      @new_formula_problems = []
      @text = FormulaText.new(formula.path)
      @specs = %w[stable head].map { |s| formula.send(s) }.compact
      @spdx_license_data = options[:spdx_license_data]
      @spdx_exception_data = options[:spdx_exception_data]
    end

    def audit_style
      return unless @style_offenses

      @style_offenses.each do |offense|
        correction_status = "#{Tty.green}[Corrected]#{Tty.reset} " if offense.corrected?

        cop_name = "#{offense.cop_name}: " if @display_cop_names
        message = "#{cop_name}#{correction_status}#{offense.message}"

        problem message, location: offense.location
      end
    end

    def audit_file
      if formula.core_formula? && @versioned_formula
        unversioned_formula = begin
          # build this ourselves as we want e.g. homebrew/core to be present
          full_name = if formula.tap
            "#{formula.tap}/#{formula.name}"
          else
            formula.name
          end
          Formulary.factory(full_name.gsub(/@.*$/, "")).path
        rescue FormulaUnavailableError, TapFormulaAmbiguityError,
               TapFormulaWithOldnameAmbiguityError
          Pathname.new formula.path.to_s.gsub(/@.*\.rb$/, ".rb")
        end
        unless unversioned_formula.exist?
          unversioned_name = unversioned_formula.basename(".rb")
          problem "#{formula} is versioned but no #{unversioned_name} formula exists"
        end
      elsif @build_stable && formula.stable? &&
            !(versioned_formulae = formula.versioned_formulae).empty?
        versioned_aliases = formula.aliases.grep(/.@\d/)
        _, last_alias_version = versioned_formulae.map(&:name).last.split("@")
        alias_name_major = "#{formula.name}@#{formula.version.major}"
        alias_name_major_minor = "#{alias_name_major}.#{formula.version.minor}"
        alias_name = if last_alias_version.split(".").length == 1
          alias_name_major
        else
          alias_name_major_minor
        end
        valid_alias_names = [alias_name_major, alias_name_major_minor]

        unless @core_tap
          versioned_aliases.map! { |a| "#{formula.tap}/#{a}" }
          valid_alias_names.map! { |a| "#{formula.tap}/#{a}" }
        end

        # Fix naming based on what people expect.
        if alias_name_major_minor == "adoptopenjdk@1.8"
          valid_alias_names << "adoptopenjdk@8"
          valid_alias_names.delete "adoptopenjdk@1"
        end

        valid_versioned_aliases = versioned_aliases & valid_alias_names
        invalid_versioned_aliases = versioned_aliases - valid_alias_names

        if valid_versioned_aliases.empty?
          if formula.tap
            problem <<~EOS
              Formula has other versions so create a versioned alias:
                cd #{formula.tap.alias_dir}
                ln -s #{formula.path.to_s.gsub(formula.tap.path, "..")} #{alias_name}
            EOS
          else
            problem "Formula has other versions so create an alias named #{alias_name}."
          end
        end

        if invalid_versioned_aliases.present?
          problem <<~EOS
            Formula has invalid versioned aliases:
              #{invalid_versioned_aliases.join("\n  ")}
          EOS
        end
      end
    end

    def self.aliases
      # core aliases + tap alias names + tap alias full name
      @aliases ||= Formula.aliases + Formula.tap_aliases
    end

    def audit_formula_name
      return unless @strict
      return unless @core_tap

      name = formula.name

      problem "'#{name}' is not allowed in homebrew/core." if MissingFormula.disallowed_reason(name)

      if Formula.aliases.include? name
        problem "Formula name conflicts with existing aliases in homebrew/core."
        return
      end

      if oldname = CoreTap.instance.formula_renames[name]
        problem "'#{name}' is reserved as the old name of #{oldname} in homebrew/core."
        return
      end

      return if formula.core_formula?
      return unless Formula.core_names.include?(name)

      problem "Formula name conflicts with existing core formula."
    end

    PROVIDED_BY_MACOS_DEPENDS_ON_ALLOWLIST = %w[
      apr
      apr-util
      libressl
      openblas
      openssl@1.1
    ].freeze

    PERMITTED_LICENSE_MISMATCHES = {
      "AGPL-3.0" => ["AGPL-3.0-only", "AGPL-3.0-or-later"],
      "GPL-2.0"  => ["GPL-2.0-only",  "GPL-2.0-or-later"],
      "GPL-3.0"  => ["GPL-3.0-only",  "GPL-3.0-or-later"],
      "LGPL-2.1" => ["LGPL-2.1-only", "LGPL-2.1-or-later"],
      "LGPL-3.0" => ["LGPL-3.0-only", "LGPL-3.0-or-later"],
    }.freeze

    PERMITTED_FORMULA_LICENSE_MISMATCHES = {
      "cmockery" => "0.1.2",
      "scw@1"    => "1.20",
    }.freeze

    def audit_license
      if formula.license.present?
        licenses, exceptions = SPDX.parse_license_expression formula.license

        non_standard_licenses = licenses.reject { |license| SPDX.valid_license? license }
        if non_standard_licenses.present?
          problem <<~EOS
            Formula #{formula.name} contains non-standard SPDX licenses: #{non_standard_licenses}.
            For a list of valid licenses check: #{Formatter.url("https://spdx.org/licenses/")}
          EOS
        end

        if @strict
          deprecated_licenses = licenses.select do |license|
            SPDX.deprecated_license? license
          end
          if deprecated_licenses.present?
            problem <<~EOS
              Formula #{formula.name} contains deprecated SPDX licenses: #{deprecated_licenses}.
              You may need to add `-only` or `-or-later` for GNU licenses (e.g. `GPL`, `LGPL`, `AGPL`, `GFDL`).
              For a list of valid licenses check: #{Formatter.url("https://spdx.org/licenses/")}
            EOS
          end
        end

        invalid_exceptions = exceptions.reject { |exception| SPDX.valid_license_exception? exception }
        if invalid_exceptions.present?
          problem <<~EOS
            Formula #{formula.name} contains invalid or deprecated SPDX license exceptions: #{invalid_exceptions}.
            For a list of valid license exceptions check:
              #{Formatter.url("https://spdx.org/licenses/exceptions-index.html")}
          EOS
        end

        return unless @online

        user, repo = get_repo_data(%r{https?://github\.com/([^/]+)/([^/]+)/?.*})
        return if user.blank?

        github_license = GitHub.get_repo_license(user, repo)
        return unless github_license
        return if (licenses + ["NOASSERTION"]).include?(github_license)
        return if PERMITTED_LICENSE_MISMATCHES[github_license]&.any? { |license| licenses.include? license }
        return if PERMITTED_FORMULA_LICENSE_MISMATCHES[formula.name] == formula.version

        problem "Formula license #{licenses} does not match GitHub license #{Array(github_license)}."

      elsif @new_formula && @core_tap
        problem "Formulae in homebrew/core must specify a license."
      end
    end

    def audit_deps
      @specs.each do |spec|
        # Check for things we don't like to depend on.
        # We allow non-Homebrew installs whenever possible.
        spec.deps.each do |dep|
          begin
            dep_f = dep.to_formula
          rescue TapFormulaUnavailableError
            # Don't complain about missing cross-tap dependencies
            next
          rescue FormulaUnavailableError
            problem "Can't find dependency #{dep.name.inspect}."
            next
          rescue TapFormulaAmbiguityError
            problem "Ambiguous dependency #{dep.name.inspect}."
            next
          rescue TapFormulaWithOldnameAmbiguityError
            problem "Ambiguous oldname dependency #{dep.name.inspect}."
            next
          end

          if dep_f.oldname && dep.name.split("/").last == dep_f.oldname
            problem "Dependency '#{dep.name}' was renamed; use new name '#{dep_f.name}'."
          end

          if self.class.aliases.include?(dep.name) &&
             dep_f.core_formula? && !dep_f.versioned_formula?
            problem "Dependency '#{dep.name}' from homebrew/core is an alias; " \
            "use the canonical name '#{dep.to_formula.full_name}'."
          end

          if @core_tap &&
             @new_formula &&
             dep_f.keg_only? &&
             dep_f.keg_only_reason.provided_by_macos? &&
             dep_f.keg_only_reason.applicable? &&
             !PROVIDED_BY_MACOS_DEPENDS_ON_ALLOWLIST.include?(dep.name)
            new_formula_problem(
              "Dependency '#{dep.name}' is provided by macOS; " \
              "please replace 'depends_on' with 'uses_from_macos'.",
            )
          end

          dep.options.each do |opt|
            next if @core_tap
            next if dep_f.option_defined?(opt)
            next if dep_f.requirements.find do |r|
              if r.recommended?
                opt.name == "with-#{r.name}"
              elsif r.optional?
                opt.name == "without-#{r.name}"
              end
            end

            problem "Dependency #{dep} does not define option #{opt.name.inspect}"
          end

          problem "Don't use git as a dependency (it's always available)" if @new_formula && dep.name == "git"

          problem "Dependency '#{dep.name}' is marked as :run. Remove :run; it is a no-op." if dep.tags.include?(:run)

          next unless @core_tap

          if dep.tags.include?(:recommended) || dep.tags.include?(:optional)
            problem "Formulae in homebrew/core should not have optional or recommended dependencies"
          end
        end

        next unless @core_tap

        if spec.requirements.map(&:recommended?).any? || spec.requirements.map(&:optional?).any?
          problem "Formulae in homebrew/core should not have optional or recommended requirements"
        end
      end
    end

    def audit_conflicts
      formula.conflicts.each do |c|
        Formulary.factory(c.name)
      rescue TapFormulaUnavailableError
        # Don't complain about missing cross-tap conflicts.
        next
      rescue FormulaUnavailableError
        problem "Can't find conflicting formula #{c.name.inspect}."
      rescue TapFormulaAmbiguityError, TapFormulaWithOldnameAmbiguityError
        problem "Ambiguous conflicting formula #{c.name.inspect}."
      end
    end

    def audit_postgresql
      return unless formula.name == "postgresql"
      return unless @core_tap

      major_version = formula.version.major.to_i
      previous_major_version = major_version - 1
      previous_formula_name = "postgresql@#{previous_major_version}"
      begin
        Formula[previous_formula_name]
      rescue FormulaUnavailableError
        problem "Versioned #{previous_formula_name} in homebrew/core must be created for " \
                "`brew-postgresql-upgrade-database` and `pg_upgrade` to work."
      end
    end

    # openssl@1.1 only needed for Linux
    VERSIONED_KEG_ONLY_ALLOWLIST = %w[
      autoconf@2.13
      bash-completion@2
      clang-format@8
      gnupg@1.4
      libsigc++@2
      lua@5.1
      numpy@1.16
      openssl@1.1
      python@3.8
    ].freeze

    def audit_versioned_keg_only
      return unless @versioned_formula
      return unless @core_tap

      if formula.keg_only?
        return if formula.keg_only_reason.versioned_formula?
        if formula.name.start_with?("openssl", "libressl") &&
           formula.keg_only_reason.by_macos?
          return
        end
      end

      return if VERSIONED_KEG_ONLY_ALLOWLIST.include?(formula.name)
      return if formula.name.start_with?("adoptopenjdk@")
      return if formula.name.start_with?("gcc@")

      problem "Versioned formulae in homebrew/core should use `keg_only :versioned_formula`"
    end

    CERT_ERROR_ALLOWLIST = {
      "hashcat"     => "https://hashcat.net/hashcat/",
      "jinx"        => "https://www.jinx-lang.org/",
      "lmod"        => "https://www.tacc.utexas.edu/research-development/tacc-projects/lmod",
      "micropython" => "https://www.micropython.org/",
      "monero"      => "https://www.getmonero.org/",
    }.freeze

    def audit_homepage
      homepage = formula.homepage

      return if homepage.nil? || homepage.empty?

      return unless @online

      return if CERT_ERROR_ALLOWLIST[formula.name] == homepage

      return unless DevelopmentTools.curl_handles_most_https_certificates?

      if http_content_problem = curl_check_http_content(homepage,
                                                        user_agents:   [:browser, :default],
                                                        check_content: true,
                                                        strict:        @strict)
        problem http_content_problem
      end
    end

    def audit_bottle_spec
      # special case: new versioned formulae should be audited
      return unless @new_formula_inclusive
      return unless @core_tap

      return if formula.bottle_disabled?

      return unless formula.bottle_defined?

      new_formula_problem "New formulae in homebrew/core should not have a `bottle do` block"
    end

    def audit_bottle_disabled
      return unless formula.bottle_disabled?
      return if formula.bottle_unneeded?

      problem "Unrecognized bottle modifier" unless formula.bottle_disable_reason.valid?

      return unless @core_tap

      problem "Formulae in homebrew/core should not use `bottle :disabled`"
    end

    def audit_github_repository_archived
      return if formula.deprecated?

      user, repo = get_repo_data(%r{https?://github\.com/([^/]+)/([^/]+)/?.*}) if @online
      return if user.blank?

      metadata = SharedAudits.github_repo_data(user, repo)
      return if metadata.nil?

      problem "GitHub repo is archived" if metadata["archived"]
    end

    def audit_gitlab_repository_archived
      return if formula.deprecated?

      user, repo = get_repo_data(%r{https?://gitlab\.com/([^/]+)/([^/]+)/?.*}) if @online
      return if user.blank?

      metadata = SharedAudits.gitlab_repo_data(user, repo)
      return if metadata.nil?

      problem "GitLab repo is archived" if metadata["archived"]
    end

    def audit_github_repository
      user, repo = get_repo_data(%r{https?://github\.com/([^/]+)/([^/]+)/?.*}) if @new_formula

      return if user.blank?

      warning = SharedAudits.github(user, repo)
      return if warning.nil?

      new_formula_problem warning
    end

    def audit_gitlab_repository
      user, repo = get_repo_data(%r{https?://gitlab\.com/([^/]+)/([^/]+)/?.*}) if @new_formula
      return if user.blank?

      warning = SharedAudits.gitlab(user, repo)
      return if warning.nil?

      new_formula_problem warning
    end

    def audit_bitbucket_repository
      user, repo = get_repo_data(%r{https?://bitbucket\.org/([^/]+)/([^/]+)/?.*}) if @new_formula
      return if user.blank?

      warning = SharedAudits.bitbucket(user, repo)
      return if warning.nil?

      new_formula_problem warning
    end

    def get_repo_data(regex)
      return unless @core_tap
      return unless @online

      _, user, repo = *regex.match(formula.stable.url) if formula.stable
      _, user, repo = *regex.match(formula.homepage) unless user
      _, user, repo = *regex.match(formula.head.url) if !user && formula.head
      return if !user || !repo

      repo.delete_suffix!(".git")

      [user, repo]
    end

    VERSIONED_HEAD_SPEC_ALLOWLIST = %w[
      bash-completion@2
      imagemagick@6
    ].freeze

    THROTTLED_FORMULAE = {
      "aws-sdk-cpp" => 10,
      "awscli@1"    => 10,
      "balena-cli"  => 10,
      "gatsby-cli"  => 10,
      "quicktype"   => 10,
      "vim"         => 50,
    }.freeze

    UNSTABLE_ALLOWLIST = {
      "aalib"           => "1.4rc",
      "automysqlbackup" => "3.0-rc",
      "aview"           => "1.3.0rc",
      "elm-format"      => "0.6.0-alpha",
      "ftgl"            => "2.1.3-rc",
      "hidapi"          => "0.8.0-rc",
      "libcaca"         => "0.99b",
      "premake"         => "4.4-beta",
      "pwnat"           => "0.3-beta",
      "recode"          => "3.7-beta",
      "speexdsp"        => "1.2rc",
      "sqoop"           => "1.4.",
      "tcptraceroute"   => "1.5beta",
      "tiny-fugue"      => "5.0b",
      "vbindiff"        => "3.0_beta",
    }.freeze

    # used for formulae that are unstable but need CI run without being in homebrew/core
    UNSTABLE_DEVEL_ALLOWLIST = {
      "python@3.9" => "3.9.0rc",
    }.freeze

    GNOME_DEVEL_ALLOWLIST = {
      "libart"              => "2.3",
      "gtk-mac-integration" => "2.1",
      "gtk-doc"             => "1.31",
      "gcab"                => "1.3",
      "libepoxy"            => "1.5",
    }.freeze

    def audit_specs
      problem "Head-only (no stable download)" if head_only?(formula)

      %w[Stable HEAD].each do |name|
        spec_name = name.downcase.to_sym
        next unless spec = formula.send(spec_name)

        ra = ResourceAuditor.new(spec, spec_name, online: @online, strict: @strict).audit
        ra.problems.each do |message|
          problem "#{name}: #{message}"
        end

        spec.resources.each_value do |resource|
          problem "Resource name should be different from the formula name" if resource.name == formula.name

          ra = ResourceAuditor.new(resource, spec_name, online: @online, strict: @strict).audit
          ra.problems.each do |message|
            problem "#{name} resource #{resource.name.inspect}: #{message}"
          end
        end

        next if spec.patches.empty?
        next unless @new_formula

        new_formula_problem(
          "Formulae should not require patches to build. " \
          "Patches should be submitted and accepted upstream first.",
        )
      end

      if stable = formula.stable
        version = stable.version
        problem "Stable: version (#{version}) is set to a string without a digit" if version.to_s !~ /\d/
        if version.to_s.start_with?("HEAD")
          problem "Stable: non-HEAD version name (#{version}) should not begin with HEAD"
        end
      end

      return unless @core_tap

      if formula.head && @versioned_formula
        head_spec_message = "Versioned formulae should not have a `HEAD` spec"
        problem head_spec_message unless VERSIONED_HEAD_SPEC_ALLOWLIST.include?(formula.name)
      end

      stable = formula.stable
      return unless stable
      return unless stable.url

      stable_version_string = stable.version.to_s
      stable_url_version = Version.parse(stable.url)
      stable_url_minor_version = stable_url_version.minor.to_i

      formula_suffix = stable.version.patch.to_i
      throttled_rate = THROTTLED_FORMULAE[formula.name]
      if throttled_rate && formula_suffix.modulo(throttled_rate).nonzero?
        problem "should only be updated every #{throttled_rate} releases on multiples of #{throttled_rate}"
      end

      case (url = stable.url)
      when /[\d._-](alpha|beta|rc\d)/
        matched = Regexp.last_match(1)
        version_prefix = stable_version_string.sub(/\d+$/, "")
        return if UNSTABLE_ALLOWLIST[formula.name] == version_prefix
        return if UNSTABLE_DEVEL_ALLOWLIST[formula.name] == version_prefix

        problem "Stable version URLs should not contain #{matched}"
      when %r{download\.gnome\.org/sources}, %r{ftp\.gnome\.org/pub/GNOME/sources}i
        version_prefix = stable.version.major_minor
        return if GNOME_DEVEL_ALLOWLIST[formula.name] == version_prefix
        return if stable_url_version < Version.create("1.0")
        return if stable_url_minor_version.even?

        problem "#{stable.version} is a development release"
      when %r{isc.org/isc/bind\d*/}i
        return if stable_url_minor_version.even?

        problem "#{stable.version} is a development release"

      when %r{https?://gitlab\.com/([\w-]+)/([\w-]+)}
        owner = Regexp.last_match(1)
        repo = Regexp.last_match(2)

        tag = SharedAudits.gitlab_tag_from_url(url)
        tag ||= stable.specs[:tag]
        tag ||= stable.version

        if @online
          error = SharedAudits.gitlab_release(owner, repo, tag, formula: formula)
          problem error if error
        end
      when %r{^https://github.com/([\w-]+)/([\w-]+)}
        owner = Regexp.last_match(1)
        repo = Regexp.last_match(2)
        tag = SharedAudits.github_tag_from_url(url)
        tag ||= formula.stable.specs[:tag]

        if @online
          error = SharedAudits.github_release(owner, repo, tag, formula: formula)
          problem error if error
        end
      end
    end

    def audit_revision_and_version_scheme
      return unless @git
      return unless formula.tap # skip formula not from core or any taps
      return unless formula.tap.git? # git log is required
      return if formula.stable.blank?

      fv = FormulaVersions.new(formula)

      current_version = formula.stable.version
      current_checksum = formula.stable.checksum
      current_version_scheme = formula.version_scheme
      current_revision = formula.revision

      previous_version = nil
      previous_version_scheme = nil
      previous_revision = nil

      newest_committed_version = nil
      newest_committed_checksum = nil
      newest_committed_revision = nil

      fv.rev_list("origin/master") do |rev|
        fv.formula_at_revision(rev) do |f|
          stable = f.stable
          next if stable.blank?

          previous_version = stable.version
          previous_checksum = stable.checksum
          previous_version_scheme = f.version_scheme
          previous_revision = f.revision

          newest_committed_version ||= previous_version
          newest_committed_checksum ||= previous_checksum
          newest_committed_revision ||= previous_revision
        end

        break if previous_version && current_version != previous_version
      end

      if current_version == previous_version &&
         current_checksum != newest_committed_checksum
        problem(
          "stable sha256 changed without the version also changing; " \
          "please create an issue upstream to rule out malicious " \
          "circumstances and to find out why the file changed.",
        )
      end

      if !newest_committed_version.nil? &&
         current_version < newest_committed_version &&
         current_version_scheme == previous_version_scheme
        problem "stable version should not decrease (from #{newest_committed_version} to #{current_version})"
      end

      unless previous_version_scheme.nil?
        if current_version_scheme < previous_version_scheme
          problem "version_scheme should not decrease (from #{previous_version_scheme} " \
                  "to #{current_version_scheme})"
        elsif current_version_scheme > (previous_version_scheme + 1)
          problem "version_schemes should only increment by 1"
        end
      end

      if (previous_version != newest_committed_version ||
         current_version != newest_committed_version) &&
         !current_revision.zero? &&
         current_revision == newest_committed_revision &&
         current_revision == previous_revision
        problem "'revision #{current_revision}' should be removed"
      elsif current_version == previous_version &&
            !previous_revision.nil? &&
            current_revision < previous_revision
        problem "revision should not decrease (from #{previous_revision} to #{current_revision})"
      elsif newest_committed_revision &&
            current_revision > (newest_committed_revision + 1)
        problem "revisions should only increment by 1"
      end
    end

    def audit_text
      bin_names = Set.new
      bin_names << formula.name
      bin_names += formula.aliases
      [formula.bin, formula.sbin].each do |dir|
        next unless dir.exist?

        bin_names += dir.children.map(&:basename).map(&:to_s)
      end
      shell_commands = ["system", "shell_output", "pipe_output"]
      bin_names.each do |name|
        shell_commands.each do |cmd|
          if text.to_s.match?(/test do.*#{cmd}[(\s]+['"]#{Regexp.escape(name)}[\s'"]/m)
            problem %Q(fully scope test #{cmd} calls, e.g. #{cmd} "\#{bin}/#{name}")
          end
        end
      end
    end

    def audit_reverse_migration
      # Only enforce for new formula being re-added to core
      return unless @strict
      return unless @core_tap
      return unless formula.tap.tap_migrations.key?(formula.name)

      problem <<~EOS
        #{formula.name} seems to be listed in tap_migrations.json!
        Please remove #{formula.name} from present tap & tap_migrations.json
        before submitting it to Homebrew/homebrew-#{formula.tap.repo}.
      EOS
    end

    def audit_prefix_has_contents
      return unless formula.prefix.directory?
      return unless Keg.new(formula.prefix).empty_installation?

      problem <<~EOS
        The installation seems to be empty. Please ensure the prefix
        is set correctly and expected files are installed.
        The prefix configure/make argument may be case-sensitive.
      EOS
    end

    def quote_dep(dep)
      dep.is_a?(Symbol) ? dep.inspect : "'#{dep}'"
    end

    def problem_if_output(output)
      problem(output) if output
    end

    def audit
      only_audits = @only
      except_audits = @except

      methods.map(&:to_s).grep(/^audit_/).each do |audit_method_name|
        name = audit_method_name.delete_prefix("audit_")
        if only_audits
          next unless only_audits.include?(name)
        elsif except_audits
          next if except_audits.include?(name)
        end
        send(audit_method_name)
      end
    end

    private

    def problem(message, location: nil)
      @problems << ({ message: message, location: location })
    end

    def new_formula_problem(message, location: nil)
      @new_formula_problems << ({ message: message, location: location })
    end

    def head_only?(formula)
      formula.head && formula.stable.nil?
    end
  end

  class ResourceAuditor
    attr_reader :name, :version, :checksum, :url, :mirrors, :using, :specs, :owner, :spec_name, :problems

    def initialize(resource, spec_name, options = {})
      @name     = resource.name
      @version  = resource.version
      @checksum = resource.checksum
      @url      = resource.url
      @mirrors  = resource.mirrors
      @using    = resource.using
      @specs    = resource.specs
      @owner    = resource.owner
      @spec_name = spec_name
      @online    = options[:online]
      @strict    = options[:strict]
      @problems  = []
    end

    def audit
      audit_version
      audit_download_strategy
      audit_urls
      self
    end

    def audit_version
      if version.nil?
        problem "missing version"
      elsif !version.detected_from_url?
        version_text = version
        version_url = Version.detect(url, **specs)
        if version_url.to_s == version_text.to_s && version.instance_of?(Version)
          problem "version #{version_text} is redundant with version scanned from URL"
        end
      end
    end

    def audit_download_strategy
      url_strategy = DownloadStrategyDetector.detect(url)

      if (using == :git || url_strategy == GitDownloadStrategy) && specs[:tag] && !specs[:revision]
        problem "Git should specify :revision when a :tag is specified."
      end

      return unless using

      if using == :cvs
        mod = specs[:module]

        problem "Redundant :module value in URL" if mod == name

        if url.match?(%r{:[^/]+$})
          mod = url.split(":").last

          if mod == name
            problem "Redundant CVS module appended to URL"
          else
            problem "Specify CVS module as `:module => \"#{mod}\"` instead of appending it to the URL"
          end
        end
      end

      return unless url_strategy == DownloadStrategyDetector.detect("", using)

      problem "Redundant :using value in URL"
    end

    def self.curl_openssl_and_deps
      @curl_openssl_and_deps ||= begin
        formulae_names = ["curl", "openssl"]
        formulae_names += formulae_names.flat_map do |f|
          Formula[f].recursive_dependencies.map(&:name)
        end
        formulae_names.uniq
      rescue FormulaUnavailableError
        []
      end
    end

    def audit_urls
      return unless @online

      urls = [url] + mirrors
      urls.each do |url|
        next if !@strict && mirrors.include?(url)

        strategy = DownloadStrategyDetector.detect(url, using)
        if strategy <= CurlDownloadStrategy && !url.start_with?("file")
          # A `brew mirror`'ed URL is usually not yet reachable at the time of
          # pull request.
          next if url.match?(%r{^https://dl.bintray.com/homebrew/mirror/})

          if http_content_problem = curl_check_http_content(url)
            problem http_content_problem
          end
        elsif strategy <= GitDownloadStrategy
          problem "The URL #{url} is not a valid git URL" unless Utils::Git.remote_exists? url
        elsif strategy <= SubversionDownloadStrategy
          next unless DevelopmentTools.subversion_handles_most_https_certificates?
          next unless Utils::Svn.available?

          problem "The URL #{url} is not a valid svn URL" unless Utils::Svn.remote_exists? url
        end
      end
    end

    def problem(text)
      @problems << text
    end
  end
end
