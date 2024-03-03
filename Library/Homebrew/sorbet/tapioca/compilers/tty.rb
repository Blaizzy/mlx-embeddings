# typed: strict
# frozen_string_literal: true

require_relative "../../../global"
require_relative "../../../utils/tty"

module Tapioca
  module Compilers
    class Tty < Tapioca::Dsl::Compiler
      ConstantType = type_member { { fixed: Module } }

      sig { override.returns(T::Enumerable[Module]) }
      def self.gather_constants
       [::Tty]
      end

      sig { override.void }
      def decorate
        root.create_path(constant) do |klass|
          dynamic_methods = ::Tty::COLOR_CODES.keys + ::Tty::STYLE_CODES.keys + ::Tty::SPECIAL_CODES.keys
          methods = ::Tty.methods(false).sort.select { |method| dynamic_methods.include?(method) }

          methods.each do |method|
            return_type = (method.to_s.end_with?("?") ? "T::Boolean" : "String")
            klass.create_method(method.to_s, return_type:)
          end
        end
      end
    end
  end
end
