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
    end
  end
end
