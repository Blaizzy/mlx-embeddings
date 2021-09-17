# frozen_string_literal: true

module Unparser
  # Allows objects to be made immutable
  #
  # Original code before vendoring and reduction from: https://github.com/dkubb/adamantium.
  module Adamantium
    module InstanceMethods
      # A noop #dup for immutable objects
      #
      # @return [self]
      #
      # @api public
      def dup
        self
      end

      # Freeze the object
      #
      # @return [Object]
      #
      # @api public
      def freeze
        memoized_method_cache
        super()
      end

    private

      def memoized_method_cache
        @memoized_method_cache ||= Memory.new({})
      end

    end # InstanceMethods

    # Storage for memoized methods
    class Memory

      # Initialize the memory storage for memoized methods
      #
      # @return [undefined]
      #
      # @api private
      def initialize(values)
        @values  = values
        @monitor = Monitor.new
        freeze
      end

      # Fetch the value from memory, or evaluate if it does not exist
      #
      # @param [Symbol] name
      #
      # @yieldreturn [Object]
      #   the value to memoize
      #
      # @api public
      def fetch(name)
        @values.fetch(name) do      # check for the key
          @monitor.synchronize do   # acquire a lock if the key is not found
            @values.fetch(name) do  # recheck under lock
              @values[name] = yield # set the value
            end
          end
        end
      end
    end # Memory

    # Methods mixed in to adamantium classes
    module ClassMethods

      # Instantiate a new frozen object
      #
      # @return [Object]
      #
      # @api public
      def new(*)
        super.freeze
      end

    end # ClassMethods

    # Methods mixed in to adamantium modules
    module ModuleMethods

      # Memoize a list of methods
      #
      # @param [Array<#to_s>] methods
      #   a list of methods to memoize
      #
      # @return [self]
      #
      # @api public
      def memoize(*methods)
        methods.each(&method(:memoize_method))
        self
      end

      # Test if method is memoized
      #
      # @param [Symbol] name
      #
      # @return [Bool]
      def memoized?(method_name)
        memoized_methods.key?(method_name)
      end

      # Return unmemoized instance method
      #
      # @param [Symbol] name
      #
      # @return [UnboundMethod]
      #   the memoized method
      #
      # @raise [NameError]
      #   raised if the method is unknown
      #
      # @api public
      def unmemoized_instance_method(method_name)
        memoized_methods.fetch(method_name) do
          fail ArgumentError, "##{method_name} is not memoized"
        end
      end

    private

      def memoize_method(method_name)
        if memoized_methods.key?(method_name)
          fail ArgumentError, "##{method_name} is already memoized"
        end

        memoized_methods[method_name] = MethodBuilder.new(self, method_name).call
      end

      def memoized_methods
        @memoized_methods ||= {}
      end

    end # ModuleMethods

    def self.included(descendant)
      descendant.class_eval do
        include InstanceMethods
        extend ModuleMethods
        extend ClassMethods if instance_of?(Class)
      end
    end
    private_class_method :included
  end # Adamantium
end # Unparser
