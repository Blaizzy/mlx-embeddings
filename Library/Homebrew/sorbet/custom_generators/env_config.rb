# typed: true
# frozen_string_literal: true

require_relative "../../global"
require_relative "../../env_config"

def nilable?(method)
  %w[browser editor github_api_token].include?(method)
end

File.open("#{File.dirname(__FILE__)}/../../env_config.rbi", "w") do |file|
  file.write(<<~RUBY)
    # typed: strict

    module Homebrew::EnvConfig
  RUBY

  dynamic_methods = {}
  Homebrew::EnvConfig::ENVS.each do |env, hash|
    next if Homebrew::EnvConfig::CUSTOM_IMPLEMENTATIONS.include?(env.to_s)

    name = Homebrew::EnvConfig.env_method_name(env, hash)
    dynamic_methods[name] = { default: hash[:default] }
  end

  methods = Homebrew::EnvConfig.methods(false).map(&:to_s).sort.select { |method| dynamic_methods.key?(method) }

  methods.each do |method|
    return_type = if method.end_with?("?")
      T::Boolean
    elsif dynamic_methods[method][:default].instance_of?(Integer)
      Integer
    else
      nilable?(method) ? T.nilable(String) : String
    end

    file.write(<<-RUBY)
  sig { returns(#{return_type}) }
  def self.#{method}; end
    RUBY

    file.write("\n") unless methods.last == method
  end

  file.write("end\n")
end
