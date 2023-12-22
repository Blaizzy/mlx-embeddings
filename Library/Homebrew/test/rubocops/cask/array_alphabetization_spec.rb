# frozen_string_literal: true

require "rubocops/rubocop-cask"

describe RuboCop::Cop::Cask::ArrayAlphabetization, :config do
  it "reports an offense when a single `zap trash` path is specified in an array" do
    expect_offense(<<~CASK)
      cask "foo" do
        url "https://example.com/foo.zip"

        zap trash: ["~/Library/Application Support/Foo"]
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Remove the `[]` around a single `zap trash` path
      end
    CASK

    expect_correction(<<~CASK)
      cask "foo" do
        url "https://example.com/foo.zip"

        zap trash: "~/Library/Application Support/Foo"
      end
    CASK
  end

  it "reports an offense when the `zap trash` paths are not in alphabetical order" do
    expect_offense(<<~CASK)
      cask "foo" do
        url "https://example.com/foo.zip"

        zap trash: [
                   ^ The `zap trash` paths should be in alphabetical order
          "/Library/Application Support/Foo",
          "/Library/Application Support/Baz",
          "~/Library/Application Support/Foo",
          "~/.dotfiles/thing",
          "~/Library/Application Support/Bar",
        ]
      end
    CASK

    expect_correction(<<~CASK)
      cask "foo" do
        url "https://example.com/foo.zip"

        zap trash: [
          "/Library/Application Support/Baz",
          "/Library/Application Support/Foo",
          "~/.dotfiles/thing",
          "~/Library/Application Support/Bar",
          "~/Library/Application Support/Foo",
        ]
      end
    CASK
  end

  it "autocorrects alphabetization in zap trash paths with interpolation" do
    expect_offense(<<~CASK)
      cask "foo" do
        url "https://example.com/foo.zip"

        zap trash: [
                   ^ The `zap trash` paths should be in alphabetical order
          "~/Library/Application Support/Foo",
          "~/Library/Application Support/Bar\#{version.major}",
        ]
      end
    CASK

    expect_correction(<<~CASK)
      cask "foo" do
        url "https://example.com/foo.zip"

        zap trash: [
          "~/Library/Application Support/Bar\#{version.major}",
          "~/Library/Application Support/Foo",
        ]
      end
    CASK
  end
end
