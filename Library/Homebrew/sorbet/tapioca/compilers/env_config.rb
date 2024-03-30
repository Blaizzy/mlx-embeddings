# typed: strict
# frozen_string_literal: true

require_relative "../../../global"
require "env_config"

module Tapioca
  module Compilers
    class EnvConfig < Tapioca::Dsl::Compiler
      # FIXME: Enable cop again when https://github.com/sorbet/sorbet/issues/3532 is fixed.
      # rubocop:disable Style/MutableConstant
      ConstantType = type_member { { fixed: Module } }
      # rubocop:enable Style/MutableConstant

      sig { override.returns(T::Enumerable[Module]) }
      def self.gather_constants = [Homebrew::EnvConfig]

      sig { override.void }
      def decorate
        root.create_module(T.must(constant.name)) do |mod|
          dynamic_methods = {}
          Homebrew::EnvConfig::ENVS.each do |env, hash|
            next if Homebrew::EnvConfig::CUSTOM_IMPLEMENTATIONS.include?(env)

            name = Homebrew::EnvConfig.env_method_name(env, hash)
            dynamic_methods[name] = hash[:default]
          end

          dynamic_methods.each do |method, default|
            return_type = if method.end_with?("?")
              T::Boolean
            elsif default
              default.class
            else
              T.nilable(String)
            end

            mod.create_method(method, return_type:, class_method: true)
          end
        end
      end
    end
  end
end
