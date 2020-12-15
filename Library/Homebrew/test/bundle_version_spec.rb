# typed: false
# frozen_string_literal: true

require "bundle_version"

describe Homebrew::BundleVersion do
  describe "#nice_version" do
    expected_mappings = {
      ["1.2", nil]            => "1.2",
      [nil, "1.2.3"]          => "1.2.3",
      ["1.2", "1.2.3"]        => "1.2.3",
      ["1.2.3", "1.2"]        => "1.2.3",
      ["1.2.3", "8312"]       => "1.2.3,8312",
      ["2021", "2006"]        => "2021,2006",
      ["1.0", "1"]            => "1.0",
      ["1.0", "0"]            => "1.0",
      ["1.2.3.4000", "4000"]  => "1.2.3.4000",
      ["5", "5.0.45"]         => "5.0.45",
      ["2.5.2(3329)", "3329"] => "2.5.2,3329",
    }

    expected_mappings.each do |(short_version, version), expected_version|
      it "maps (#{short_version.inspect}, #{version.inspect}) to #{expected_version.inspect}" do
        expect(described_class.new(short_version, version).nice_version)
          .to eq expected_version
      end
    end
  end
end
