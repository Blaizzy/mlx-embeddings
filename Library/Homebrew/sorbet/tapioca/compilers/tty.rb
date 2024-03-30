# typed: strict
# frozen_string_literal: true

require_relative "../../../global"
require "utils/tty"

module Tapioca
  module Compilers
    class Tty < Tapioca::Dsl::Compiler
      # FIXME: Enable cop again when https://github.com/sorbet/sorbet/issues/3532 is fixed.
      # rubocop:disable Style/MutableConstant
      ConstantType = type_member { { fixed: Module } }
      # rubocop:enable Style/MutableConstant

      sig { override.returns(T::Enumerable[Module]) }
      def self.gather_constants = [::Tty]

      sig { override.void }
      def decorate
        root.create_module(T.must(constant.name)) do |mod|
          dynamic_methods = ::Tty::COLOR_CODES.keys + ::Tty::STYLE_CODES.keys + ::Tty::SPECIAL_CODES.keys

          dynamic_methods.each do |method|
            mod.create_method(method.to_s, return_type: "String", class_method: true)
          end
        end
      end
    end
  end
end
