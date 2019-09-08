# frozen_string_literal: true

describe Cask::Blacklist, :cask do
  describe "::blacklisted_reason" do
    matcher :blacklist do |name|
      match do |expected|
        expected.blacklisted_reason(name)
      end
    end

    it { is_expected.not_to blacklist("adobe-air") }
    it { is_expected.to blacklist("adobe-after-effects") }
    it { is_expected.to blacklist("adobe-illustrator") }
    it { is_expected.to blacklist("adobe-indesign") }
    it { is_expected.to blacklist("adobe-photoshop") }
    it { is_expected.to blacklist("adobe-premiere") }
    it { is_expected.to blacklist("audacity") }
    it { is_expected.to blacklist("pharo") }
    it { is_expected.not_to blacklist("non-blacklisted-cask") }
  end
end
