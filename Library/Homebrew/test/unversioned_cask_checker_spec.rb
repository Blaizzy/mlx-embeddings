# typed: false
# frozen_string_literal: true

require "unversioned_cask_checker"

describe Homebrew::UnversionedCaskChecker do
  describe "::decide_between_versions" do
    expected_mappings = {
      [nil, nil]             => nil,
      ["1.2", nil]           => "1.2",
      [nil, "1.2.3"]         => "1.2.3",
      ["1.2", "1.2.3"]       => "1.2.3",
      ["1.2.3", "1.2"]       => "1.2.3",
      ["1.2.3", "8312"]      => "1.2.3,8312",
      ["2021", "2006"]       => "2021,2006",
      ["1.0", "1"]           => "1.0",
      ["1.0", "0"]           => "1.0",
      ["1.2.3.4000", "4000"] => "1.2.3.4000",
      ["5", "5.0.45"]        => "5.0.45",
    }

    expected_mappings.each do |(short_version, version), expected_version|
      it "maps (#{short_version}, #{version}) to #{expected_version}" do
        expect(described_class.decide_between_versions(short_version, version))
          .to eq expected_version
      end
    end
  end
end
