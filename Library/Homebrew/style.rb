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
      raise ArgumentError, "Invalid output type: #{output_type.inspect}" unless [:print, :json].include?(output_type)

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
        run_shellcheck(shell_files, output_type)
      end

      if output_type == :json
        Offenses.new(rubocop_result + shellcheck_result)
      else
        rubocop_result && shellcheck_result
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

      require "rubocops"

      args = %w[
        --force-exclusion
      ]
      args << if fix
        "-A"
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

      has_non_formula = files.any? do |file|
        File.expand_path(file).start_with? HOMEBREW_LIBRARY_PATH
      end

      if files.any? && !has_non_formula
        config = if files.first && File.exist?("#{files.first}/spec")
          HOMEBREW_LIBRARY/".rubocop_rspec.yml"
        else
          HOMEBREW_LIBRARY/".rubocop.yml"
        end
        args << "--config" << config
      end

      if files.blank?
        args << HOMEBREW_LIBRARY_PATH
      else
        args += files
      end

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

    def run_shellcheck(files, output_type)
      shellcheck   = Formula["shellcheck"].opt_bin/"shellcheck" if Formula["shellcheck"].any_version_installed?
      shellcheck ||= which("shellcheck")
      shellcheck ||= which("shellcheck", ENV["HOMEBREW_PATH"])
      shellcheck ||= begin
        ohai "Installing `shellcheck` for shell style checks..."
        safe_system HOMEBREW_BREW_FILE, "install", "shellcheck"
        Formula["shellcheck"].opt_bin/"shellcheck"
      end

      if files.empty?
        files = [
          HOMEBREW_BREW_FILE,
          # TODO: HOMEBREW_REPOSITORY/"completions/bash/brew",
          *Pathname.glob("#{HOMEBREW_LIBRARY}/Homebrew/*.sh"),
          *Pathname.glob("#{HOMEBREW_LIBRARY}/Homebrew/cmd/*.sh"),
          *Pathname.glob("#{HOMEBREW_LIBRARY}/Homebrew/utils/*.sh"),
        ]
      end

      args = ["--shell=bash", "--", *files] # TODO: Add `--enable=all` to check for more problems.

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

    def json_result!(result)
      # An exit status of 1 just means violations were found; other numbers mean
      # execution errors.
      # JSON needs to be at least 2 characters.
      result.assert_success! if !(0..1).cover?(result.status.exitstatus) || result.stdout.length < 2

      JSON.parse(result.stdout)
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
