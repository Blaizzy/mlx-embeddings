# frozen_string_literal: true

module Cask
  module Blacklist
    def self.blacklisted_reason(name)
      case name
      when /^adobe\-(after|illustrator|indesign|photoshop|premiere)/
        "Adobe casks were removed because they are too difficult to maintain."
      when /^audacity$/
        "Audacity was removed because it is too difficult to download programmatically."
      when /^pharo$/
        "Pharo developers maintain their own tap."
      end
    end
  end
end
