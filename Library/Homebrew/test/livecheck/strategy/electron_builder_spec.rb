# typed: false
# frozen_string_literal: true

require "livecheck/strategy"

describe Homebrew::Livecheck::Strategy::ElectronBuilder do
  subject(:electron_builder) { described_class }

  let(:yaml_url) { "https://www.example.com/example/latest-mac.yml" }
  let(:non_yaml_url) { "https://brew.sh/test" }

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

  let(:versions) { ["1.2.3"] }

  describe "::match?" do
    it "returns true for a YAML file URL" do
      expect(electron_builder.match?(yaml_url)).to be true
    end

    it "returns false for non-YAML URL" do
      expect(electron_builder.match?(non_yaml_url)).to be false
    end
  end

  describe "::versions_from_content" do
    it "returns an empty array if content is blank" do
      expect(electron_builder.versions_from_content("")).to eq([])
    end

    it "returns an array of version strings when given YAML text" do
      expect(electron_builder.versions_from_content(electron_builder_yaml)).to eq(versions)
    end

    it "returns an array of version strings when given YAML text and a block" do
      # Returning a string from block
      expect(
        electron_builder.versions_from_content(electron_builder_yaml) do |yaml|
          yaml["version"].sub("3", "4")
        end,
      ).to eq(["1.2.4"])

      # Returning an array of strings from block
      expect(electron_builder.versions_from_content(electron_builder_yaml) { versions }).to eq(versions)
    end

    it "allows a nil return from a block" do
      expect(electron_builder.versions_from_content(electron_builder_yaml) { next }).to eq([])
    end

    it "errors on an invalid return type from a block" do
      expect { electron_builder.versions_from_content(electron_builder_yaml) { 123 } }
        .to raise_error(TypeError, Homebrew::Livecheck::Strategy::INVALID_BLOCK_RETURN_VALUE_MSG)
    end
  end
end
