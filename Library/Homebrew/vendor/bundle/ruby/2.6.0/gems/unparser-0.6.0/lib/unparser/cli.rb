# frozen_string_literal: true

module Unparser
  # Unparser CLI implementation
  class CLI

    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1

    class Target
      include AbstractType

      # Path target
      class Path < self
        include Concord.new(:path)

        # Validation for this target
        #
        # @return [Validation]
        def validation
          Validation.from_path(path)
        end

        # Literal for this target
        #
        # @return [Validation]
        def literal_validation
          Validation::Literal.from_path(path)
        end
      end

      # String target
      class String
        include Concord.new(:string)

        # Validation for this target
        #
        # @return [Validation]
        def validation
          Validation.from_string(string)
        end

        # Literal for this target
        #
        # @return [Validation]
        def literal_validation
          Validation::Literal.from_string(path)
        end
      end # String
    end # Target

    private_constant(*constants(false))

    # Run CLI
    #
    # @param [Array<String>] arguments
    #
    # @return [Integer]
    #   the exit status
    #
    # @api private
    #
    def self.run(*arguments)
      new(*arguments).exit_status
    end

    # Initialize object
    #
    # @param [Array<String>] arguments
    #
    # @return [undefined]
    #
    # @api private
    def initialize(arguments)
      @ignore  = Set.new
      @targets = []

      @fail_fast  = false
      @success    = true
      @validation = :validation
      @verbose    = false

      opts = OptionParser.new do |builder|
        add_options(builder)
      end

      opts.parse!(arguments).each do |name|
        @targets.concat(targets(name))
      end
    end

    # Add options
    #
    # @param [OptionParser] builder
    #
    # @return [undefined]
    #
    # @api private
    #
    # rubocop:disable Metrics/MethodLength
    def add_options(builder)
      builder.banner = 'usage: unparse [options] FILE [FILE]'
      builder.separator('')
      builder.on('-e', '--evaluate SOURCE') do |source|
        @targets << Target::String.new(source)
      end
      builder.on('--start-with FILE') do |path|
        @start_with = targets(path).first
      end
      builder.on('-v', '--verbose') do
        @verbose = true
      end
      builder.on('-l', '--literal') do
        @validation = :literal_validation
      end
      builder.on('--ignore FILE') do |file|
        @ignore.merge(targets(file))
      end
      builder.on('--fail-fast') do
        @fail_fast = true
      end
    end
    # rubocop:enable Metrics/MethodLength

    # Return exit status
    #
    # @return [Integer]
    #
    # @api private
    #
    def exit_status
      effective_targets.each do |target|
        process_target(target)
        break if @fail_fast && !@success
      end

      @success ? EXIT_SUCCESS : EXIT_FAILURE
    end

  private

    def process_target(target)
      validation = target.public_send(@validation)
      if validation.success?
        puts validation.report if @verbose
        puts "Success: #{validation.identification}"
      else
        puts validation.report
        puts "Error: #{validation.identification}"
        @success = false
      end
    end

    def effective_targets
      if @start_with
        reject = true
        @targets.reject do |targets|
          if reject && targets.eql?(@start_with)
            reject = false
          end

          reject
        end
      else
        @targets
      end.reject(&@ignore.method(:include?))
    end

    def targets(file_name)
      if File.directory?(file_name)
        Dir.glob(File.join(file_name, '**/*.rb')).sort
      elsif File.file?(file_name)
        [file_name]
      else
        Dir.glob(file_name).sort
      end.map { |file| Target::Path.new(Pathname.new(file)) }
    end
  end # CLI
end # Unparser
