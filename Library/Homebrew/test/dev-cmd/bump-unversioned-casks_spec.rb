# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"
require "dev-cmd/bump-unversioned-casks"

describe "Homebrew.bump_unversioned_casks_args" do
  it_behaves_like "parseable arguments"

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
    }

    expected_mappings.each do |(short_version, version), expected_version|
      it "maps (#{short_version}, #{version}) to #{expected_version}" do
        expect(Homebrew.decide_between_versions(short_version, version)).to eq expected_version
      end
    end
  end
end
