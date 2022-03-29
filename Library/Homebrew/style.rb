# typed: false
# frozen_string_literal: true

require "shellwords"

module Homebrew
  # Helper module for running RuboCop.
  #
  # @api private
  module Style
    module_function

    # Checks style for a list of files, printing simple RuboCop output.
    # Returns true if violations were found, false otherwise.
    def check_style_and_print(files, **options)
      success = check_style_impl(files, :print, **options)

      if ENV["GITHUB_ACTIONS"] && !success
        check_style_json(files, **options).each do |path, offenses|
          offenses.each do |o|
            line = o.location.line
            column = o.location.line

            annotation = GitHub::Actions::Annotation.new(:error, o.message, file: path, line: line, column: column)
            puts annotation if annotation.relevant?
          end
        end
      end

      success
    end

    # Checks style for a list of files, returning results as an {Offenses}
    # object parsed from its JSON output.
    def check_style_json(files, **options)
      check_style_impl(files, :json, **options)
    end

    def check_style_impl(files, output_type,
                         fix: false,
                         except_cops: nil, only_cops: nil,
                         display_cop_names: false,
                         reset_cache: false,
                         debug: false, verbose: false)
      raise ArgumentError, "Invalid output type: #{output_type.inspect}" if [:print, :json].exclude?(output_type)

      shell_files, ruby_files =
        Array(files).map(&method(:Pathname))
                    .partition { |f| f.realpath == HOMEBREW_BREW_FILE.realpath || f.extname == ".sh" }

      rubocop_result = if shell_files.any? && ruby_files.none?
        output_type == :json ? [] : true
      else
        run_rubocop(ruby_files, output_type,
                    fix: fix,
                    except_cops: except_cops, only_cops: only_cops,
                    display_cop_names: display_cop_names,
                    reset_cache: reset_cache,
                    debug: debug, verbose: verbose)
      end

      shellcheck_result = if ruby_files.any? && shell_files.none?
        output_type == :json ? [] : true
      else
        run_shellcheck(shell_files, output_type, fix: fix)
      end

      shfmt_result = if ruby_files.any? && shell_files.none?
        true
      else
        run_shfmt(shell_files, fix: fix)
      end

      if output_type == :json
        Offenses.new(rubocop_result + shellcheck_result)
      else
        rubocop_result && shellcheck_result && shfmt_result
      end
    end

    RUBOCOP = (HOMEBREW_LIBRARY_PATH/"utils/rubocop.rb").freeze

    def run_rubocop(files, output_type,
                    fix: false, except_cops: nil, only_cops: nil, display_cop_names: false, reset_cache: false,
                    debug: false, verbose: false)
      Homebrew.install_bundler_gems!

      require "warnings"

      Warnings.ignore :parser_syntax do
        require "rubocop"
      end

      require "rubocops/all"

      args = %w[
        --force-exclusion
      ]
      args << if fix
        "--auto-correct-all"
      else
        "--parallel"
      end

      args += ["--extra-details"] if verbose
      args += ["--display-cop-names"] if display_cop_names || verbose

      if except_cops
        except_cops.map! { |cop| RuboCop::Cop::Cop.registry.qualified_cop_name(cop.to_s, "") }
        cops_to_exclude = except_cops.select do |cop|
          RuboCop::Cop::Cop.registry.names.include?(cop) ||
            RuboCop::Cop::Cop.registry.departments.include?(cop.to_sym)
        end

        args << "--except" << cops_to_exclude.join(",") unless cops_to_exclude.empty?
      elsif only_cops
        only_cops.map! { |cop| RuboCop::Cop::Cop.registry.qualified_cop_name(cop.to_s, "") }
        cops_to_include = only_cops.select do |cop|
          RuboCop::Cop::Cop.registry.names.include?(cop) ||
            RuboCop::Cop::Cop.registry.departments.include?(cop.to_sym)
        end

        odie "RuboCops #{only_cops.join(",")} were not found" if cops_to_include.empty?

        args << "--only" << cops_to_include.join(",")
      end

      files&.map!(&:expand_path)
      if files.blank? || files == [HOMEBREW_REPOSITORY]
        files = [HOMEBREW_LIBRARY_PATH]
      elsif files.none? { |f| f.to_s.start_with? HOMEBREW_LIBRARY_PATH }
        config = if files.any? { |f| (f/"spec").exist? }
          HOMEBREW_LIBRARY/".rubocop_rspec.yml"
        else
          HOMEBREW_LIBRARY/".rubocop.yml"
        end
        args << "--config" << config
      end

      args += files

      cache_env = { "XDG_CACHE_HOME" => "#{HOMEBREW_CACHE}/style" }

      FileUtils.rm_rf cache_env["XDG_CACHE_HOME"] if reset_cache

      case output_type
      when :print
        args << "--debug" if debug

        # Don't show the default formatter's progress dots
        # on CI or if only checking a single file.
        args << "--format" << "clang" if ENV["CI"] || files.count { |f| !f.directory? } == 1

        args << "--color" if Tty.color?

        system cache_env, RUBY_PATH, RUBOCOP, *args
        $CHILD_STATUS.success?
      when :json
        result = system_command RUBY_PATH,
                                args: [RUBOCOP, "--format", "json", *args],
                                env:  cache_env
        json = json_result!(result)
        json["files"]
      end
    end

    def run_shellcheck(files, output_type, fix: false)
      files = shell_scripts if files.blank?

      files = files.map(&:realpath) # use absolute file paths

      args = [
        "--shell=bash",
        "--enable=all",
        "--external-sources",
        "--source-path=#{HOMEBREW_LIBRARY}",
        "--",
        *files,
      ]

      if fix
        # patch options:
        #   -g 0 (--get=0)       : suppress environment variable `PATCH_GET`
        #   -f   (--force)       : we know what we are doing, force apply patches
        #   -d / (--directory=/) : change to root directory, since we use absolute file paths
        #   -p0  (--strip=0)     : do not strip path prefixes, since we are at root directory
        # NOTE: we use short flags where for compatibility
        patch_command = %w[patch -g 0 -f -d / -p0]
        patches = system_command(shellcheck, args: ["--format=diff", *args]).stdout
        Utils.safe_popen_write(*patch_command) { |p| p.write(patches) } if patches.present?
      end

      case output_type
      when :print
        system shellcheck, "--format=tty", *args
        $CHILD_STATUS.success?
      when :json
        result = system_command shellcheck, args: ["--format=json", *args]
        json = json_result!(result)

        # Convert to same format as RuboCop offenses.
        severity_hash = { "style" => "refactor", "info" => "convention" }
        json.group_by { |v| v["file"] }
            .map do |k, v|
          {
            "path"     => k,
            "offenses" => v.map do |o|
              o.delete("file")

              o["cop_name"] = "SC#{o.delete("code")}"

              level = o.delete("level")
              o["severity"] = severity_hash.fetch(level, level)

              line = o.delete("line")
              column = o.delete("column")

              o["corrected"] = false
              o["correctable"] = o.delete("fix").present?

              o["location"] = {
                "start_line"   => line,
                "start_column" => column,
                "last_line"    => o.delete("endLine"),
                "last_column"  => o.delete("endColumn"),
                "line"         => line,
                "column"       => column,
              }

              o
            end,
          }
        end
      end
    end

    def run_shfmt(files, fix: false)
      files = shell_scripts if files.blank?
      # Do not format completions and Dockerfile
      files.delete(HOMEBREW_REPOSITORY/"completions/bash/brew")
      files.delete(HOMEBREW_REPOSITORY/"Dockerfile")

      # shfmt options:
      #   -i 2     : indent by 2 spaces
      #   -ci      : indent switch cases
      #   -ln bash : language variant to parse ("bash")
      #   -w       : write result to file instead of stdout (inplace fixing)
      # "--" is needed for `utils/shfmt.sh`
      args = ["-i", "2", "-ci", "-ln", "bash", "--", *files]

      # Do inplace fixing
      args.unshift("-w") if fix # need to add before "--"

      system shfmt, *args
      $CHILD_STATUS.success?
    end

    def json_result!(result)
      # An exit status of 1 just means violations were found; other numbers mean
      # execution errors.
      # JSON needs to be at least 2 characters.
      result.assert_success! if !(0..1).cover?(result.status.exitstatus) || result.stdout.length < 2

      JSON.parse(result.stdout)
    end

    def shell_scripts
      [
        HOMEBREW_BREW_FILE,
        HOMEBREW_REPOSITORY/"completions/bash/brew",
        HOMEBREW_REPOSITORY/"Dockerfile",
        *HOMEBREW_LIBRARY.glob("Homebrew/*.sh"),
        *HOMEBREW_LIBRARY.glob("Homebrew/shims/**/*").map(&:realpath).uniq
                         .reject { |path| path.directory? || path.basename.to_s == "cc" },
        *HOMEBREW_LIBRARY.glob("Homebrew/{dev-,}cmd/*.sh"),
        *HOMEBREW_LIBRARY.glob("Homebrew/{cask/,}utils/*.sh"),
      ]
    end

    def shellcheck
      ensure_formula_installed!("shellcheck", latest: true,
                                              reason: "shell style checks").opt_bin/"shellcheck"
    end

    def shfmt
      ensure_formula_installed!("shfmt", latest: true,
                                         reason: "formatting shell scripts")
      HOMEBREW_LIBRARY/"Homebrew/utils/shfmt.sh"
    end

    # Collection of style offenses.
    class Offenses
      include Enumerable

      def initialize(paths)
        @offenses = {}
        paths.each do |f|
          next if f["offenses"].empty?

          path = Pathname(f["path"]).realpath
          @offenses[path] = f["offenses"].map { |x| Offense.new(x) }
        end
      end

      def for_path(path)
        @offenses.fetch(Pathname(path), [])
      end

      def each(*args, &block)
        @offenses.each(*args, &block)
      end
    end

    # A style offense.
    class Offense
      attr_reader :severity, :message, :corrected, :location, :cop_name

      def initialize(json)
        @severity = json["severity"]
        @message = json["message"]
        @cop_name = json["cop_name"]
        @corrected = json["corrected"]
        @location = LineLocation.new(json["location"])
      end

      def severity_code
        @severity[0].upcase
      end

      def corrected?
        @corrected
      end
    end

    # Source location of a style offense.
    class LineLocation
      extend T::Sig

      attr_reader :line, :column

      def initialize(json)
        @line = json["line"]
        @column = json["column"]
      end

      sig { returns(String) }
      def to_s
        "#{line}: col #{column}"
      end
    end
  end
end
