# typed: strict
# frozen_string_literal: true

require "cask/cmd/abstract_command"
require "cask/cmd/info"
require "cask/cask_loader"
require "cask/caskroom"

module Homebrew
  module MissingFormula
    extend T::Sig
    class << self
      extend T::Sig
      sig { params(name: String).returns(T.nilable(String)) }
      def disallowed_reason(name)
        case name.downcase
        when "xcode"
          <<~EOS
            Xcode can be installed from the App Store.
          EOS
        when "tex", "tex-live", "texlive", "mactex", "latex"
          <<~EOS
            There are three versions of MacTeX.

            Full installation:
              brew install --cask mactex

            Full installation without bundled applications:
              brew install --cask mactex-no-gui

            Minimal installation:
              brew install --cask basictex
          EOS
        else
          generic_disallowed_reason(name)
        end
      end

      sig { params(name: String, silent: T::Boolean, show_info: T::Boolean).returns(T.nilable(String)) }
      def cask_reason(name, silent: false, show_info: false)
        return if silent

        suggest_command(name, show_info ? "info" : "install")
      end

      sig { params(name: String, command: String).returns(T.nilable(String)) }
      def suggest_command(name, command)
        suggestion = <<~EOS
          Found a cask named "#{name}" instead. Try
            brew #{command} --cask #{name}

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
