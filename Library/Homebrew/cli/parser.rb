# frozen_string_literal: true

require "cli/args"
require "optparse"
require "set"

COMMAND_DESC_WIDTH = 80
OPTION_DESC_WIDTH = 43

module Homebrew
  module CLI
    class Parser
      attr_reader :processed_options, :hide_from_man_page

      def self.parse(args = ARGV, &block)
        new(args, &block).parse(args)
      end

      def self.global_options
        {
          quiet:   [["-q", "--quiet"], :quiet, "Suppress any warnings."],
          verbose: [["-v", "--verbose"], :verbose, "Make some output more verbose."],
          debug:   [["-d", "--debug"], :debug, "Display any debugging information."],
          force:   [["-f", "--force"], :force, "Override warnings and enable potentially unsafe operations."],
        }
      end

      def initialize(args = ARGV, &block)
        @parser = OptionParser.new
        @args = Homebrew::CLI::Args.new(argv: ARGV_WITHOUT_MONKEY_PATCHING)
        @args[:remaining] = []
        @args[:cmdline_args] = args.dup
        @constraints = []
        @conflicts = []
        @switch_sources = {}
        @processed_options = []
        @max_named_args = nil
        @hide_from_man_page = false
        instance_eval(&block)
        post_initialize
      end

      def post_initialize
        @parser.on_tail("-h", "--help", "Show this message.") do
          puts generate_help_text
          exit 0
        end
      end

      def switch(*names, description: nil, env: nil, required_for: nil, depends_on: nil)
        global_switch = names.first.is_a?(Symbol)
        names, env, default_description = common_switch(*names) if global_switch
        if description.nil? && global_switch
          description = default_description
        elsif description.nil?
          description = option_to_description(*names)
        end
        process_option(*names, description)
        @parser.on(*names, *wrap_option_desc(description)) do
          enable_switch(*names, from: :args)
        end

        names.each do |name|
          set_constraints(name, required_for: required_for, depends_on: depends_on)
        end

        enable_switch(*names, from: :env) if !env.nil? && !ENV["HOMEBREW_#{env.to_s.upcase}"].nil?
      end
      alias switch_option switch

      def usage_banner(text)
        @parser.banner = Formatter.wrap("#{text}\n", COMMAND_DESC_WIDTH)
      end

      def usage_banner_text
        @parser.banner
      end

      def comma_array(name, description: nil)
        description = option_to_description(name) if description.nil?
        process_option(name, description)
        @parser.on(name, OptionParser::REQUIRED_ARGUMENT, Array, *wrap_option_desc(description)) do |list|
          @args[option_to_name(name)] = list
        end
      end

      def flag(*names, description: nil, required_for: nil, depends_on: nil)
        if names.any? { |name| name.end_with? "=" }
          required = OptionParser::REQUIRED_ARGUMENT
        else
          required = OptionParser::OPTIONAL_ARGUMENT
        end
        names.map! { |name| name.chomp "=" }
        description = option_to_description(*names) if description.nil?
        process_option(*names, description)
        @parser.on(*names, *wrap_option_desc(description), required) do |option_value|
          names.each do |name|
            @args[option_to_name(name)] = option_value
          end
        end

        names.each do |name|
          set_constraints(name, required_for: required_for, depends_on: depends_on)
        end
      end

      def conflicts(*options)
        @conflicts << options.map { |option| option_to_name(option) }
      end

      def option_to_name(option)
        option.sub(/\A--?/, "")
              .tr("-", "_")
              .delete("=")
      end

      def name_to_option(name)
        if name.length == 1
          "-#{name}"
        else
          "--#{name.tr("_", "-")}"
        end
      end

      def option_to_description(*names)
        names.map { |name| name.to_s.sub(/\A--?/, "").tr("-", " ") }.max
      end

      def summary
        @parser.to_s
      end

      def parse(cmdline_args = ARGV)
        raise "Arguments were already parsed!" if @args_parsed

        begin
          remaining_args = @parser.parse(cmdline_args)
        rescue OptionParser::InvalidOption => e
          $stderr.puts generate_help_text
          raise e
        end
        check_constraint_violations
        check_named_args(remaining_args)
        @args[:remaining] = remaining_args
        @args.freeze_processed_options!(@processed_options)
        Homebrew.args = @args
        cmdline_args.freeze
        @args_parsed = true
        @parser
      end

      def global_option?(name, desc)
        Homebrew::CLI::Parser.global_options.key?(name.to_sym) &&
          Homebrew::CLI::Parser.global_options[name.to_sym].last == desc
      end

      def generate_help_text
        @parser.to_s.sub(/^/, "#{Tty.bold}Usage: brew#{Tty.reset} ")
               .gsub(/`(.*?)`/m, "#{Tty.bold}\\1#{Tty.reset}")
               .gsub(%r{<([^\s]+?://[^\s]+?)>}) { |url| Formatter.url(url) }
               .gsub(/<(.*?)>/m, "#{Tty.underline}\\1#{Tty.reset}")
               .gsub(/\*(.*?)\*/m, "#{Tty.underline}\\1#{Tty.reset}")
      end

      def formula_options
        @args.formulae.each do |f|
          next if f.options.empty?

          f.options.each do |o|
            name = o.flag
            description = "`#{f.name}`: #{o.description}"
            if name.end_with? "="
              flag   name, description: description
            else
              switch name, description: description
            end
          end
        end
      rescue FormulaUnavailableError
        []
      end

      def max_named(count)
        @max_named_args = count
      end

      def hide_from_man_page!
        @hide_from_man_page = true
      end

      private

      def enable_switch(*names, from:)
        names.each do |name|
          @switch_sources[option_to_name(name)] = from
          @args["#{option_to_name(name)}?"] = true
        end
      end

      def disable_switch(*names)
        names.each do |name|
          @args.delete_field("#{option_to_name(name)}?")
        end
      end

      # These are common/global switches accessible throughout Homebrew
      def common_switch(name)
        Homebrew::CLI::Parser.global_options.fetch(name, name)
      end

      def option_passed?(name)
        @args.respond_to?(name) || @args.respond_to?("#{name}?")
      end

      def wrap_option_desc(desc)
        Formatter.wrap(desc, OPTION_DESC_WIDTH).split("\n")
      end

      def set_constraints(name, depends_on:, required_for:)
        secondary = option_to_name(name)
        unless required_for.nil?
          primary = option_to_name(required_for)
          @constraints << [primary, secondary, :mandatory]
        end

        return if depends_on.nil?

        primary = option_to_name(depends_on)
        @constraints << [primary, secondary, :optional]
      end

      def check_constraints
        @constraints.each do |primary, secondary, constraint_type|
          primary_passed = option_passed?(primary)
          secondary_passed = option_passed?(secondary)
          if :mandatory.equal?(constraint_type) && primary_passed && !secondary_passed
            raise OptionConstraintError.new(primary, secondary)
          end
          raise OptionConstraintError.new(primary, secondary, missing: true) if secondary_passed && !primary_passed
        end
      end

      def check_conflicts
        @conflicts.each do |mutually_exclusive_options_group|
          violations = mutually_exclusive_options_group.select do |option|
            option_passed? option
          end

          next if violations.count < 2

          env_var_options = violations.select do |option|
            @switch_sources[option_to_name(option)] == :env
          end

          select_cli_arg = violations.count - env_var_options.count == 1
          raise OptionConflictError, violations.map(&method(:name_to_option)) unless select_cli_arg

          env_var_options.each(&method(:disable_switch))
        end
      end

      def check_invalid_constraints
        @conflicts.each do |mutually_exclusive_options_group|
          @constraints.each do |p, s|
            next unless Set[p, s].subset?(Set[*mutually_exclusive_options_group])

            raise InvalidConstraintError.new(p, s)
          end
        end
      end

      def check_constraint_violations
        check_invalid_constraints
        check_conflicts
        check_constraints
      end

      def check_named_args(args)
        raise NamedArgumentsError, @max_named_args if !@max_named_args.nil? && args.size > @max_named_args
      end

      def process_option(*args)
        option, = @parser.make_switch(args)
        @processed_options << [option.short.first, option.long.first, option.arg, option.desc.first]
      end
    end

    class OptionConstraintError < RuntimeError
      def initialize(arg1, arg2, missing: false)
        message = if !missing
          "`#{arg1}` and `#{arg2}` should be passed together."
        else
          "`#{arg2}` cannot be passed without `#{arg1}`."
        end
        super message
      end
    end

    class OptionConflictError < RuntimeError
      def initialize(args)
        args_list = args.map(&Formatter.public_method(:option))
                        .join(" and ")
        super "Options #{args_list} are mutually exclusive."
      end
    end

    class InvalidConstraintError < RuntimeError
      def initialize(arg1, arg2)
        super "`#{arg1}` and `#{arg2}` cannot be mutually exclusive and mutually dependent simultaneously."
      end
    end

    class NamedArgumentsError < UsageError
      def initialize(maximum)
        message = case maximum
        when 0
          "This command does not take named arguments."
        when 1
          "This command does not take multiple named arguments."
        else
          "This command does not take more than #{maximum} named arguments."
        end
        super message
      end
    end
  end
end
