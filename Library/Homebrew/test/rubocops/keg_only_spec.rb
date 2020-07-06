# frozen_string_literal: true

require "rubocops/keg_only"

describe RuboCop::Cop::FormulaAudit::KegOnly do
  subject(:cop) { described_class.new }

  specify "keg_only_needs_downcasing" do
    expect_offense(<<~RUBY)
      class Foo < Formula

        url "https://brew.sh/foo-1.0.tgz"
        homepage "https://brew.sh"

        keg_only "Because why not"
                 ^^^^^^^^^^^^^^^^^ 'Because' from the `keg_only` reason should be 'because'.
      end
    RUBY
  end

  specify "keg_only_redundant_period" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        homepage "https://brew.sh"

        keg_only "ending with a period."
                 ^^^^^^^^^^^^^^^^^^^^^^^ `keg_only` reason should not end with a period.
      end
    RUBY
  end

  specify "keg_only_autocorrects_downcasing" do
    source = <<~RUBY
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        homepage "https://brew.sh"
        keg_only "Because why not"
      end
    RUBY

    corrected_source = <<~RUBY
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        homepage "https://brew.sh"
        keg_only "because why not"
      end
    RUBY

    new_source = autocorrect_source(source)
    expect(new_source).to eq(corrected_source)
  end

  specify "keg_only_autocorrects_redundant_period" do
    source = <<~RUBY
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        homepage "https://brew.sh"
        keg_only "ending with a period."
      end
    RUBY

    corrected_source = <<~RUBY
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        homepage "https://brew.sh"
        keg_only "ending with a period"
      end
    RUBY

    new_source = autocorrect_source(source)
    expect(new_source).to eq(corrected_source)
  end

  specify "keg_only_handles_block_correctly" do
    expect_no_offenses(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        homepage "https://brew.sh"

        keg_only <<~EOF
          this line starts with a lowercase word.

          This line does not but that shouldn't be a
          problem
        EOF
      end
    RUBY
  end

  specify "keg_only_handles_allowlist_correctly" do
    expect_no_offenses(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        homepage "https://brew.sh"

        keg_only "Apple ships foo in the CLT package"
      end
    RUBY
  end

  specify "keg_only does not need downcasing of formula name in reason" do
    filename = Formulary.core_path("foo")
    File.open(filename, "w") do |file|
      FileUtils.chmod "-rwx", filename

      expect_no_offenses(<<~RUBY, file)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"

          keg_only "Foo is the formula name hence downcasing is not required"
        end
      RUBY
    end
  end
end
