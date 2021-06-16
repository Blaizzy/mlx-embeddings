# typed: false
# frozen_string_literal: true

require "utils/path"

describe Utils do
  describe "::path_is_parent_of?" do
    it "returns true when child path is a descendant of the parent" do
      expect(described_class.path_is_parent_of?("/foo", "/foo/bar/baz")).to eq(true)
    end

    it "returns false when child path is not a descendant of the parent" do
      expect(described_class.path_is_parent_of?("/foo/bar/baz/qux", "/foo/bar")).to eq(false)
    end
  end

  describe "::shortened_brew_path" do
    it "returns shortened path when the path can be expressed with the output of a brew command" do
      expect(described_class.shortened_brew_path("#{HOMEBREW_PREFIX}/foo/bar")).to eq("$(brew --prefix)/foo/bar")
    end

    it "returns shortened path with $(brew --prefix <formula>) when path is in a linked keg", :integration_test do
      install_test_formula "testball"
      f = Formula["testball"]

      expect(described_class.shortened_brew_path("#{f.opt_prefix}/main.c")).to eq("$(brew --prefix testball)/main.c")
    end

    it "returns the original path when the path cannot be shortened" do
      expect(described_class.shortened_brew_path("/foo/bar/baz")).to eq("/foo/bar/baz")
    end
  end
end
