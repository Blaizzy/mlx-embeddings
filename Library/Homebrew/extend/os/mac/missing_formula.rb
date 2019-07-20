# frozen_string_literal: true

require "cask/cmd/abstract_command"
require "cask/cmd/info"
require "cask/cask_loader"
require "cask/caskroom"

module Homebrew
  module MissingFormula
    class << self
      def blacklisted_reason(name)
        case name.downcase
        when "xcode"
          <<~EOS
            Xcode can be installed from the App Store.
          EOS
        else
          generic_blacklisted_reason(name)
        end
      end

      def cask_reason(name, silent: false, show_info: false)
        return if silent

        suggest_command(name, show_info ? "info" : "install")
      end

      def suggest_command(name, command)
        suggestion = <<~EOS
          Found a cask named "#{name}" instead. Try
            brew cask #{command} #{name}

        EOS
        case command
        when "install"
          Cask::CaskLoader.load(name)
        when "uninstall"
          cask = Cask::Caskroom.casks.find { |installed_cask| installed_cask.to_s == name }
          raise Cask::CaskUnavailableError, name if cask.nil?
        when "info"
          cask = Cask::CaskLoader.load(name)
          suggestion = <<~EOS
            Found a cask named "#{name}" instead.

            #{Cask::Cmd::Info.get_info(cask)}
          EOS
        else
          return
        end
        suggestion
      rescue Cask::CaskUnavailableError
        nil
      end
    end
  end
end
