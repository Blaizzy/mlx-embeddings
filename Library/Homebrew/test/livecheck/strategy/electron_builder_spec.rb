# typed: false
# frozen_string_literal: true

require "livecheck/strategy/electron_builder"

describe Homebrew::Livecheck::Strategy::ElectronBuilder do
  subject(:electron_builder) { described_class }

  let(:valid_url) { "https://www.example.com/example/latest-mac.yml" }
  let(:invalid_url) { "https://brew.sh/test" }

  let(:electron_builder_yaml) {
    <<~EOS
      version: 1.2.3
      files:
        - url: Example-1.2.3-mac.zip
          sha512: MDXR0pxozBJjxxbtUQJOnhiaiiQkryLAwtcVjlnNiz30asm/PtSxlxWKFYN3kV/kl+jriInJrGypuzajTF6XIA==
          size: 92031237
          blockMapSize: 96080
        - url: Example-1.2.3.dmg
          sha512: k6WRDlZEfZGZHoOfUShpHxXZb5p44DRp+FAO2FXNx2kStZvyW9VuaoB7phPMfZpcMKrzfRfncpP8VEM8OB2y9g==
          size: 94972630
      path: Example-1.2.3-mac.zip
      sha512: MDXR0pxozBJjxxbtUQJOnhiaiiQkryLAwtcVjlnNiz30asm/PtSxlxWKFYN3kV/kl+jriInJrGypuzajTF6XIA==
      releaseDate: '2000-01-01T00:00:00.000Z'
    EOS
  }

  describe "::match?" do
    it "returns true for any URL pointing to a YAML file" do
      expect(electron_builder.match?(valid_url)).to be true
    end

    it "returns false for a URL not pointing to a YAML file" do
      expect(electron_builder.match?(invalid_url)).to be false
    end
  end

  describe "::version_from_content" do
    let(:version_from_electron_builder_yaml) { electron_builder.version_from_content(electron_builder_yaml) }

    it "returns nil if content is blank" do
      expect(electron_builder.version_from_content("")).to be nil
    end

    it "returns a version string when given YAML data" do
      expect(version_from_electron_builder_yaml).to be_a(String)
    end

    it "returns a version string when given YAML data and a block" do
      version = electron_builder.version_from_content(electron_builder_yaml) do |yaml|
        yaml["version"].sub("3", "4")
      end

      expect(version).to eq "1.2.4"
    end
  end
end
