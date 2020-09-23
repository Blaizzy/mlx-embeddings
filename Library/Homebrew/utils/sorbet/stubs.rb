# typed: false
# frozen_string_literal: true

# Stubs for `sorbet-runtime`, all taken from `sorbet/t` except for `T::Sig.sig`.
#
# @private
module T
  # rubocop:disable Style/Documentation
  module Sig
    module WithoutRuntime
      def self.sig(arg = nil, &blk); end
    end

    module_function

    def sig(arg = nil, &blk); end
  end

  def self.any(type_a, type_b, *types); end

  def self.nilable(type); end

  def self.untyped; end

  def self.noreturn; end

  def self.all(type_a, type_b, *types); end

  def self.enum(values); end

  def self.proc; end

  def self.self_type; end

  def self.class_of(klass); end

  def self.type_alias(type = nil, &blk); end

  def self.type_parameter(name); end

  def self.cast(value, _type, checked: true)
    value
  end

  def self.let(value, _type, checked: true)
    value
  end

  def self.assert_type!(value, _type, checked: true)
    value
  end

  def self.unsafe(value)
    value
  end

  def self.must(arg, _msg = nil)
    arg
  end

  def self.reveal_type(value)
    value
  end

  module Array
    def self.[](type); end
  end

  module Hash
    def self.[](keys, values); end
  end

  module Enumerable
    def self.[](type); end
  end

  module Range
    def self.[](type); end
  end

  module Set
    def self.[](type); end
  end
  # rubocop:enable Style/Documentation
end
