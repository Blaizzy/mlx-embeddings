# frozen_string_literal: true

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

        cask = Cask::CaskLoader.load(name)
        reason = +"Found a cask named \"#{name}\" instead.\n"
        if show_info
          reason << Cask::Cmd::Info.get_info(cask)
        else
          reason << "Did you mean to type \"brew cask install #{name}\"?\n"
        end
        reason.freeze
      rescue Cask::CaskUnavailableError
        nil
      end
    end
  end
end
