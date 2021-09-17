# frozen_string_literal: true

module Unparser
  # A mixin to define a composition
  #
  # Original code before vendoring and reduction from: https://github.com/mbj/concord.
  class Concord < Module
    include Adamantium, Equalizer.new(:names)

    # The maximum number of objects the hosting class is composed of
    MAX_NR_OF_OBJECTS = 3

    # Return names
    #
    # @return [Enumerable<Symbol>]
    #
    # @api private
    #
    attr_reader :names

    private

    # Initialize object
    #
    # @return [undefined]
    #
    # @api private
    #
    # rubocop:disable Lint/MissingSuper
    def initialize(*names)
      if names.length > MAX_NR_OF_OBJECTS
        fail "Composition of more than #{MAX_NR_OF_OBJECTS} objects is not allowed"
      end

      @names = names
      define_initialize
      define_readers
      define_equalizer
    end
    # rubocop:enable Lint/MissingSuper

    # Define equalizer
    #
    # @return [undefined]
    #
    # @api private
    #
    def define_equalizer
      include(Equalizer.new(*names))
    end

    # Define readers
    #
    # @return [undefined]
    #
    # @api private
    #
    def define_readers
      attribute_names = names
      attr_reader(*attribute_names)

      protected(*attribute_names) if attribute_names.any?
    end

    # Define initialize method
    #
    # @return [undefined]
    #
    # @api private
    #
    #
    def define_initialize
      ivars = instance_variable_names
      size = names.size

      define_method :initialize do |*args|
        args_size = args.size
        unless args_size.equal?(size)
          fail ArgumentError, "wrong number of arguments (#{args_size} for #{size})"
        end

        ivars.zip(args) { |ivar, arg| instance_variable_set(ivar, arg) }
      end
    end

    # Return instance variable names
    #
    # @return [String]
    #
    # @api private
    #
    def instance_variable_names
      names.map { |name| "@#{name}" }
    end

    # Mixin for public attribute readers
    class Public < self

      # Hook called when module is included
      #
      # @param [Class,Module] descendant
      #
      # @return [undefined]
      #
      # @api private
      #
      def included(descendant)
        names.each do |name|
          descendant.__send__(:public, name)
        end
      end
    end # Public
  end # Concord
end # Unparser
