# frozen_string_literal: true

module Unparser
  module Adamantium
    # Build the memoized method
    class MethodBuilder

      # Raised when the method arity is invalid
      class InvalidArityError < ArgumentError

        # Initialize an invalid arity exception
        #
        # @param [Module] descendant
        # @param [Symbol] method
        # @param [Integer] arity
        #
        # @api private
        def initialize(descendant, method, arity)
          super("Cannot memoize #{descendant}##{method}, its arity is #{arity}")
        end

      end # InvalidArityError

      # Raised when a block is passed to a memoized method
      class BlockNotAllowedError < ArgumentError

        # Initialize a block not allowed exception
        #
        # @param [Module] descendant
        # @param [Symbol] method
        #
        # @api private
        def initialize(descendant, method)
          super("Cannot pass a block to #{descendant}##{method}, it is memoized")
        end

      end # BlockNotAllowedError

      # Initialize an object to build a memoized method
      #
      # @param [Module] descendant
      # @param [Symbol] method_name
      #
      # @return [undefined]
      #
      # @api private
      def initialize(descendant, method_name)
        @descendant          = descendant
        @method_name         = method_name
        @original_visibility = visibility
        @original_method     = @descendant.instance_method(@method_name)
        assert_arity(@original_method.arity)
      end

      # Build a new memoized method
      #
      # @example
      #   method_builder.call  # => creates new method
      #
      # @return [UnboundMethod]
      #
      # @api public
      def call
        remove_original_method
        create_memoized_method
        set_method_visibility
        @original_method
      end

    private

      def assert_arity(arity)
        if arity.nonzero?
          fail InvalidArityError.new(@descendant, @method_name, arity)
        end
      end

      def remove_original_method
        name = @method_name
        @descendant.module_eval { undef_method(name) }
      end

      def create_memoized_method
        name =   @method_name
        method = @original_method
        @descendant.module_eval do
          define_method(name) do |&block|
            fail BlockNotAllowedError.new(self.class, name) if block

            memoized_method_cache.fetch(name) do
              method.bind(self).call.freeze
            end
          end
        end
      end

      def set_method_visibility
        @descendant.__send__(@original_visibility, @method_name)
      end

      def visibility
        if    @descendant.private_method_defined?(@method_name)   then :private
        elsif @descendant.protected_method_defined?(@method_name) then :protected
        else                                                           :public
        end
      end

    end # MethodBuilder
  end # Adamantium
end # Unparser
