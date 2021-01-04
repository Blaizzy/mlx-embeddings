# typed: false
# frozen_string_literal: true

require "utils/ast"

describe Utils::AST do
  let(:initial_formula) do
    <<~RUBY
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tar.gz"
        license all_of: [
          :public_domain,
          "MIT",
          "GPL-3.0-or-later" => { with: "Autoconf-exception-3.0" },
        ]
      end
    RUBY
  end

  describe ".replace_formula_stanza!" do
    it "replaces the specified stanza in a formula" do
      contents = initial_formula.dup
      described_class.replace_formula_stanza!(contents, :license, :public_domain)
      expect(contents).to eq <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tar.gz"
          license :public_domain
        end
      RUBY
    end
  end

  describe ".add_formula_stanza!" do
    it "adds the specified stanza to a formula" do
      contents = initial_formula.dup
      described_class.add_formula_stanza!(contents, :revision, 1)
      expect(contents).to eq <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tar.gz"
          license all_of: [
            :public_domain,
            "MIT",
            "GPL-3.0-or-later" => { with: "Autoconf-exception-3.0" },
          ]
          revision 1
        end
      RUBY
    end
  end

  describe ".stanza_text" do
    let(:compound_license) do
      <<~RUBY.chomp
        license all_of: [
          :public_domain,
          "MIT",
          "GPL-3.0-or-later" => { with: "Autoconf-exception-3.0" },
        ]
      RUBY
    end

    it "accepts existing stanza text" do
      expect(described_class.stanza_text(:revision, "revision 1")).to eq("revision 1")
      expect(described_class.stanza_text(:license, "license :public_domain")).to eq("license :public_domain")
      expect(described_class.stanza_text(:license, 'license "MIT"')).to eq('license "MIT"')
      expect(described_class.stanza_text(:license, compound_license)).to eq(compound_license)
    end

    it "accepts a number as the stanza value" do
      expect(described_class.stanza_text(:revision, 1)).to eq("revision 1")
    end

    it "accepts a symbol as the stanza value" do
      expect(described_class.stanza_text(:license, :public_domain)).to eq("license :public_domain")
    end

    it "accepts a string as the stanza value" do
      expect(described_class.stanza_text(:license, "MIT")).to eq('license "MIT"')
    end

    it "adds indent to stanza text if specified" do
      expect(described_class.stanza_text(:revision, "revision 1", indent: 2)).to eq("  revision 1")
      expect(described_class.stanza_text(:license, 'license "MIT"', indent: 2)).to eq('  license "MIT"')
      expect(described_class.stanza_text(:license, compound_license, indent: 2)).to eq(compound_license.indent(2))
    end

    it "does not add indent if already indented" do
      expect(described_class.stanza_text(:revision, "  revision 1", indent: 2)).to eq("  revision 1")
      expect(
        described_class.stanza_text(:license, compound_license.indent(2), indent: 2),
      ).to eq(compound_license.indent(2))
    end
  end

  describe ".add_bottle_stanza!" do
    let(:bottle_output) do
      <<~RUBY.chomp.indent(2)
        bottle do
          sha256 "f7b1fc772c79c20fddf621ccc791090bc1085fcef4da6cca03399424c66e06ca" => :sierra
        end
      RUBY
    end

    context "when `license` is a string" do
      let(:formula_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"
            license "MIT"
          end
        RUBY
      end

      let(:new_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"
            license "MIT"

            bottle do
              sha256 "f7b1fc772c79c20fddf621ccc791090bc1085fcef4da6cca03399424c66e06ca" => :sierra
            end
          end
        RUBY
      end

      it "adds `bottle` after `license`" do
        described_class.add_bottle_stanza!(formula_contents, bottle_output)
        expect(formula_contents).to eq(new_contents)
      end
    end

    context "when `license` is a symbol" do
      let(:formula_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"
            license :cannot_represent
          end
        RUBY
      end

      let(:new_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"
            license :cannot_represent

            bottle do
              sha256 "f7b1fc772c79c20fddf621ccc791090bc1085fcef4da6cca03399424c66e06ca" => :sierra
            end
          end
        RUBY
      end

      it "adds `bottle` after `license`" do
        described_class.add_bottle_stanza!(formula_contents, bottle_output)
        expect(formula_contents).to eq(new_contents)
      end
    end

    context "when `license` is multiline" do
      let(:formula_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"
            license all_of: [
              :public_domain,
              "MIT",
              "GPL-3.0-or-later" => { with: "Autoconf-exception-3.0" },
            ]
          end
        RUBY
      end

      let(:new_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"
            license all_of: [
              :public_domain,
              "MIT",
              "GPL-3.0-or-later" => { with: "Autoconf-exception-3.0" },
            ]

            bottle do
              sha256 "f7b1fc772c79c20fddf621ccc791090bc1085fcef4da6cca03399424c66e06ca" => :sierra
            end
          end
        RUBY
      end

      it "adds `bottle` after `license`" do
        described_class.add_bottle_stanza!(formula_contents, bottle_output)
        expect(formula_contents).to eq(new_contents)
      end
    end

    context "when `head` is a string" do
      let(:formula_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"
            head "https://brew.sh/foo.git"
          end
        RUBY
      end

      let(:new_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"
            head "https://brew.sh/foo.git"

            bottle do
              sha256 "f7b1fc772c79c20fddf621ccc791090bc1085fcef4da6cca03399424c66e06ca" => :sierra
            end
          end
        RUBY
      end

      it "adds `bottle` after `head`" do
        described_class.add_bottle_stanza!(formula_contents, bottle_output)
        expect(formula_contents).to eq(new_contents)
      end
    end

    context "when `head` is a block" do
      let(:formula_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"

            head do
              url "https://brew.sh/foo.git"
            end
          end
        RUBY
      end

      let(:new_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"

            bottle do
              sha256 "f7b1fc772c79c20fddf621ccc791090bc1085fcef4da6cca03399424c66e06ca" => :sierra
            end

            head do
              url "https://brew.sh/foo.git"
            end
          end
        RUBY
      end

      it "adds `bottle` before `head`" do
        described_class.add_bottle_stanza!(formula_contents, bottle_output)
        expect(formula_contents).to eq(new_contents)
      end
    end

    context "when there is a comment on the same line" do
      let(:formula_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz" # comment
          end
        RUBY
      end

      let(:new_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz" # comment

            bottle do
              sha256 "f7b1fc772c79c20fddf621ccc791090bc1085fcef4da6cca03399424c66e06ca" => :sierra
            end
          end
        RUBY
      end

      it "adds `bottle` after the comment" do
        described_class.add_bottle_stanza!(formula_contents, bottle_output)
        expect(formula_contents).to eq(new_contents)
      end
    end

    context "when the next line is a comment" do
      let(:formula_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"
            # comment
          end
        RUBY
      end

      let(:new_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"
            # comment

            bottle do
              sha256 "f7b1fc772c79c20fddf621ccc791090bc1085fcef4da6cca03399424c66e06ca" => :sierra
            end
          end
        RUBY
      end

      it "adds `bottle` after the comment" do
        described_class.add_bottle_stanza!(formula_contents, bottle_output)
        expect(formula_contents).to eq(new_contents)
      end
    end

    context "when the next line is blank and the one after it is a comment" do
      let(:formula_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"

            # comment
          end
        RUBY
      end

      let(:new_contents) do
        <<~RUBY.chomp
          class Foo < Formula
            url "https://brew.sh/foo-1.0.tar.gz"

            bottle do
              sha256 "f7b1fc772c79c20fddf621ccc791090bc1085fcef4da6cca03399424c66e06ca" => :sierra
            end

            # comment
          end
        RUBY
      end

      it "adds `bottle` before the comment" do
        described_class.add_bottle_stanza!(formula_contents, bottle_output)
        expect(formula_contents).to eq(new_contents)
      end
    end
  end
end
