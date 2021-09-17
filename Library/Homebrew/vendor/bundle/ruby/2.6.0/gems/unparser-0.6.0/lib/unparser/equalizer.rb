# frozen_string_literal: true

module Unparser
  # Define equality, equivalence and inspection methods
  #
  # Original code before vendoring and reduction from: https://github.com/dkubb/equalizer.
  class Equalizer < Module
    # Initialize an Equalizer with the given keys
    #
    # Will use the keys with which it is initialized to define #cmp?,
    # #hash, and #inspect
    #
    # @param [Array<Symbol>] keys
    #
    # @return [undefined]
    #
    # @api private
    #
    # rubocop:disable Lint/MissingSuper
    def initialize(*keys)
      @keys = keys
      define_methods
      freeze
    end
    # rubocop:enable Lint/MissingSuper

    private

    def included(descendant)
      descendant.include(Methods)
    end

    def define_methods
      define_cmp_method
      define_hash_method
      define_inspect_method
    end

    def define_cmp_method
      keys = @keys
      define_method(:cmp?) do |comparator, other|
        keys.all? do |key|
          __send__(key).public_send(comparator, other.__send__(key))
        end
      end
      private :cmp?
    end

    def define_hash_method
      keys = @keys
      define_method(:hash) do
        keys.map(&public_method(:__send__)).push(self.class).hash
      end
    end

    def define_inspect_method
      keys = @keys
      define_method(:inspect) do
        klass = self.class
        name  = klass.name || klass.inspect
        "#<#{name}#{keys.map { |key| " #{key}=#{__send__(key).inspect}" }.join}>"
      end
    end

    # The comparison methods
    module Methods
      # Compare the object with other object for equality
      #
      # @example
      #   object.eql?(other)  # => true or false
      #
      # @param [Object] other
      #   the other object to compare with
      #
      # @return [Boolean]
      #
      # @api public
      def eql?(other)
        instance_of?(other.class) && cmp?(__method__, other)
      end

      # Compare the object with other object for equivalency
      #
      # @example
      #   object == other  # => true or false
      #
      # @param [Object] other
      #   the other object to compare with
      #
      # @return [Boolean]
      #
      # @api public
      def ==(other)
        instance_of?(other.class) && cmp?(__method__, other)
      end
    end # module Methods
  end # class Equalizer
end # Unparser
