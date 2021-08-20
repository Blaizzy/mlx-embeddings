# typed: false
# frozen_string_literal: true

require "deprecate_disable"
require "formula_text_auditor"
require "resource_auditor"

module Homebrew
  # Auditor for checking common violations in {Formula}e.
  #
  # @api private
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
      @text = FormulaTextAuditor.new(formula.path)
      @specs = %w[stable head].map { |s| formula.send(s) }.compact
      @spdx_license_data = options[:spdx_license_data]
      @spdx_exception_data = options[:spdx_exception_data]
      @tap_audit_exceptions = options[:tap_audit_exceptions]
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
      elsif @build_stable &&
            formula.stable? &&
            !@versioned_formula &&
            (versioned_formulae = formula.versioned_formulae - [formula]) &&
            versioned_formulae.present?
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
      name = formula.name

      problem "Formula name '#{name}' must not contain uppercase letters." if name != name.downcase

      return unless @strict
      return unless @core_tap

      problem "'#{name}' is not allowed in homebrew/core." if MissingFormula.disallowed_reason(name)

      if Formula.aliases.include? name
        problem "Formula name conflicts with existing aliases in homebrew/core."
        return
      end

      if (oldname = CoreTap.instance.formula_renames[name])
        problem "'#{name}' is reserved as the old name of #{oldname} in homebrew/core."
        return
      end

      return if formula.core_formula?
      return unless Formula.core_names.include?(name)

      problem "Formula name conflicts with existing core formula."
    end

    PERMITTED_LICENSE_MISMATCHES = {
      "AGPL-3.0" => ["AGPL-3.0-only", "AGPL-3.0-or-later"],
      "GPL-2.0"  => ["GPL-2.0-only",  "GPL-2.0-or-later"],
      "GPL-3.0"  => ["GPL-3.0-only",  "GPL-3.0-or-later"],
      "LGPL-2.1" => ["LGPL-2.1-only", "LGPL-2.1-or-later"],
      "LGPL-3.0" => ["LGPL-3.0-only", "LGPL-3.0-or-later"],
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
        return if tap_audit_exception :permitted_formula_license_mismatches, formula.name

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
            problem "Can't find dependency '#{dep.name.inspect}'."
            next
          rescue TapFormulaAmbiguityError
            problem "Ambiguous dependency '#{dep.name.inspect}'."
            next
          rescue TapFormulaWithOldnameAmbiguityError
            problem "Ambiguous oldname dependency '#{dep.name.inspect}'."
            next
          end

          if dep_f.oldname && dep.name.split("/").last == dep_f.oldname
            problem "Dependency '#{dep.name}' was renamed; use new name '#{dep_f.name}'."
          end

          if @core_tap &&
             @new_formula &&
             dep_f.keg_only? &&
             dep_f.keg_only_reason.provided_by_macos? &&
             dep_f.keg_only_reason.applicable? &&
             formula.requirements.none?(LinuxRequirement) &&
             !tap_audit_exception(:provided_by_macos_depends_on_allowlist, dep.name)
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

            problem "Dependency '#{dep}' does not define option #{opt.name.inspect}"
          end

          problem "Don't use 'git' as a dependency (it's always available)" if @new_formula && dep.name == "git"

          problem "Dependency '#{dep.name}' is marked as :run. Remove :run; it is a no-op." if dep.tags.include?(:run)

          next unless @core_tap

          if self.class.aliases.include?(dep.name)
            problem "Dependency '#{dep.name}' is an alias; use the canonical name '#{dep.to_formula.full_name}'."
          end

          if dep.tags.include?(:recommended) || dep.tags.include?(:optional)
            problem "Formulae in homebrew/core should not have optional or recommended dependencies"
          end
        end

        next unless @core_tap

        if spec.requirements.map(&:recommended?).any? || spec.requirements.map(&:optional?).any?
          problem "Formulae in homebrew/core should not have optional or recommended requirements"
        end
      end

      return unless @core_tap
      return if tap_audit_exception :versioned_dependencies_conflicts_allowlist, formula.name

      # The number of conflicts on Linux is absurd.
      # TODO: remove this and check these there too.
      return if OS.linux? && !Homebrew::EnvConfig.simulate_macos_on_linux?

      recursive_runtime_formulae = formula.runtime_formula_dependencies(undeclared: false)
      version_hash = {}
      version_conflicts = Set.new
      recursive_runtime_formulae.each do |f|
        name = f.name
        unversioned_name, = name.split("@")
        version_hash[unversioned_name] ||= Set.new
        version_hash[unversioned_name] << name
        next if version_hash[unversioned_name].length < 2

        version_conflicts += version_hash[unversioned_name]
      end

      return if version_conflicts.empty?

      return if formula.disabled?

      return if formula.deprecated? &&
                formula.deprecation_reason != DeprecateDisable::DEPRECATE_DISABLE_REASONS[:versioned_formula]

      problem <<~EOS
        #{formula.full_name} contains conflicting version recursive dependencies:
          #{version_conflicts.to_a.join ", "}
        View these with `brew deps --tree #{formula.full_name}`.
      EOS
    end

    def audit_conflicts
      tap = formula.tap
      formula.conflicts.each do |conflict|
        conflicting_formula = Formulary.factory(conflict.name)
        next if tap != conflicting_formula.tap

        problem "Formula should not conflict with itself" if formula == conflicting_formula

        if tap.formula_renames.key?(conflict.name) || tap.aliases.include?(conflict.name)
          problem "Formula conflict should be declared using " \
                  "canonical name (#{conflicting_formula.name}) instead of #{conflict.name}"
        end

        reverse_conflict_found = false
        conflicting_formula.conflicts.each do |reverse_conflict|
          reverse_conflict_formula = Formulary.factory(reverse_conflict.name)
          if tap.formula_renames.key?(reverse_conflict.name) || tap.aliases.include?(reverse_conflict.name)
            problem "Formula #{conflicting_formula.name} conflict should be declared using " \
                    "canonical name (#{reverse_conflict_formula.name}) instead of #{reverse_conflict.name}"
          end

          reverse_conflict_found ||= reverse_conflict_formula == formula
        end
        unless reverse_conflict_found
          problem "Formula #{conflicting_formula.name} should also have a conflict declared with #{formula.name}"
        end
      rescue TapFormulaUnavailableError
        # Don't complain about missing cross-tap conflicts.
        next
      rescue FormulaUnavailableError
        problem "Can't find conflicting formula #{conflict.name.inspect}."
      rescue TapFormulaAmbiguityError, TapFormulaWithOldnameAmbiguityError
        problem "Ambiguous conflicting formula #{conflict.name.inspect}."
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
                "`brew postgresql-upgrade-database` and `pg_upgrade` to work."
      end
    end

    def audit_glibc
      return if formula.name != "glibc"
      return unless @core_tap

      version = formula.version.to_s
      return if version == OS::CI_GLIBC_VERSION

      problem "The glibc version must be #{version}, as this is the version used by our CI on Linux. " \
              "Glibc is for users who have a system Glibc with a lower version, " \
              "which allows them to use our Linux bottles, which were compiled against system Glibc on CI."
    end

    ELASTICSEARCH_KIBANA_RELICENSED_VERSION = "7.11"

    def audit_elasticsearch_kibana
      return if formula.name != "elasticsearch" && formula.name != "kibana"
      return unless @core_tap
      return if formula.version < Version.new(ELASTICSEARCH_KIBANA_RELICENSED_VERSION)

      problem "Elasticsearch and Kibana were relicensed to a non-open-source license from version 7.11. " \
              "They must not be upgraded to version 7.11 or newer."
    end

    def audit_versioned_keg_only
      return unless @versioned_formula
      return unless @core_tap

      if formula.keg_only?
        return if formula.keg_only_reason.versioned_formula?
        return if formula.name.start_with?("openssl", "libressl") && formula.keg_only_reason.by_macos?
      end

      return if tap_audit_exception :versioned_keg_only_allowlist, formula.name

      problem "Versioned formulae in homebrew/core should use `keg_only :versioned_formula`"
    end

    def audit_homepage
      homepage = formula.homepage

      return if homepage.blank?

      return unless @online

      return if tap_audit_exception :cert_error_allowlist, formula.name, homepage

      return unless DevelopmentTools.curl_handles_most_https_certificates?

      use_homebrew_curl = false
      %w[Stable HEAD].each do |name|
        spec_name = name.downcase.to_sym
        next unless (spec = formula.send(spec_name))

        use_homebrew_curl = spec.using == :homebrew_curl
        break if use_homebrew_curl
      end

      if (http_content_problem = curl_check_http_content(homepage,
                                                         "homepage URL",
                                                         user_agents:       [:browser, :default],
                                                         check_content:     true,
                                                         strict:            @strict,
                                                         use_homebrew_curl: use_homebrew_curl))
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

      if !formula.bottle_disable_reason.valid?
        problem "Unrecognized bottle modifier"
      elsif @core_tap
        if formula.bottle_unneeded?
          problem "Formulae in homebrew/core should not use `bottle :unneeded`"
        else
          problem "Formulae in homebrew/core should not use `bottle :disabled`"
        end
      end
    end

    def audit_github_repository_archived
      return if formula.deprecated? || formula.disabled?

      user, repo = get_repo_data(%r{https?://github\.com/([^/]+)/([^/]+)/?.*}) if @online
      return if user.blank?

      metadata = SharedAudits.github_repo_data(user, repo)
      return if metadata.nil?

      problem "GitHub repo is archived" if metadata["archived"]
    end

    def audit_gitlab_repository_archived
      return if formula.deprecated? || formula.disabled?

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

    def audit_specs
      problem "Head-only (no stable download)" if head_only?(formula)

      %w[Stable HEAD].each do |name|
        spec_name = name.downcase.to_sym
        next unless (spec = formula.send(spec_name))

        except = @except.to_a
        if spec_name == :head &&
           tap_audit_exception(:head_non_default_branch_allowlist, formula.name, spec.specs[:branch])
          except << "head_branch"
        end

        ra = ResourceAuditor.new(
          spec, spec_name,
          online: @online, strict: @strict, only: @only, except: except,
          use_homebrew_curl: spec.using == :homebrew_curl
        ).audit
        ra.problems.each do |message|
          problem "#{name}: #{message}"
        end

        spec.resources.each_value do |resource|
          problem "Resource name should be different from the formula name" if resource.name == formula.name

          ra = ResourceAuditor.new(
            resource, spec_name,
            online: @online, strict: @strict, only: @only, except: @except
          ).audit
          ra.problems.each do |message|
            problem "#{name} resource #{resource.name.inspect}: #{message}"
          end
        end

        next if spec.patches.empty?
        next if !@new_formula || !@core_tap

        new_formula_problem(
          "Formulae should not require patches to build. " \
          "Patches should be submitted and accepted upstream first.",
        )
      end

      return unless @core_tap

      if formula.head && @versioned_formula &&
         !tap_audit_exception(:versioned_head_spec_allowlist, formula.name)
        problem "Versioned formulae should not have a `HEAD` spec"
      end

      stable = formula.stable
      return unless stable
      return unless stable.url

      version = stable.version
      problem "Stable: version (#{version}) is set to a string without a digit" if version.to_s !~ /\d/

      stable_version_string = version.to_s
      if stable_version_string.start_with?("HEAD")
        problem "Stable: non-HEAD version name (#{stable_version_string}) should not begin with HEAD"
      end

      stable_url_version = Version.parse(stable.url)
      stable_url_minor_version = stable_url_version.minor.to_i

      formula_suffix = stable.version.patch.to_i
      throttled_rate = tap_audit_exception(:throttled_formulae, formula.name)
      if throttled_rate && formula_suffix.modulo(throttled_rate).nonzero?
        problem "should only be updated every #{throttled_rate} releases on multiples of #{throttled_rate}"
      end

      case (url = stable.url)
      when /[\d._-](alpha|beta|rc\d)/
        matched = Regexp.last_match(1)
        version_prefix = stable_version_string.sub(/\d+$/, "")
        return if tap_audit_exception :unstable_allowlist, formula.name, version_prefix
        return if tap_audit_exception :unstable_devel_allowlist, formula.name, version_prefix

        problem "Stable version URLs should not contain #{matched}"
      when %r{download\.gnome\.org/sources}, %r{ftp\.gnome\.org/pub/GNOME/sources}i
        version_prefix = stable.version.major_minor
        return if tap_audit_exception :gnome_devel_allowlist, formula.name, version_prefix
        return if stable_url_version < Version.create("1.0")
        # All minor versions are stable in the new GNOME version scheme (which starts at version 40.0)
        # https://discourse.gnome.org/t/new-gnome-versioning-scheme/4235
        return if stable_url_version >= Version.create("40.0")
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
      current_url = formula.stable.url

      previous_version = nil
      previous_version_scheme = nil
      previous_revision = nil

      newest_committed_version = nil
      newest_committed_checksum = nil
      newest_committed_revision = nil
      newest_committed_url = nil

      fv.rev_list("origin/master") do |rev|
        begin
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
            newest_committed_url ||= stable.url
          end
        rescue MacOSVersionError
          break
        end

        break if previous_version && current_version != previous_version
        break if previous_revision && current_revision != previous_revision
      end

      if current_version == newest_committed_version &&
         current_url == newest_committed_url &&
         current_checksum != newest_committed_checksum &&
         current_checksum.present? && newest_committed_checksum.present?
        problem(
          "stable sha256 changed without the url/version also changing; " \
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
        next if only_audits&.exclude?(name)
        next if except_audits&.include?(name)

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

    def tap_audit_exception(list, formula, value = nil)
      return false if @tap_audit_exceptions.blank?
      return false unless @tap_audit_exceptions.key? list

      list = @tap_audit_exceptions[list]

      case list
      when Array
        list.include? formula
      when Hash
        return false unless list.include? formula
        return list[formula] if value.blank?

        list[formula] == value
      end
    end
  end
end
