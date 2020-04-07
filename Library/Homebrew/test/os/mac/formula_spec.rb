# frozen_string_literal: true

require "formula"

describe Formula do
  describe "#on_macos" do
    it "defines an url on macos only" do
      f = formula do
        homepage "https://brew.sh"

        on_macos do
          url "https://brew.sh/test-macos-0.1.tbz"
          sha256 TEST_SHA256
        end

        on_linux do
          url "https://brew.sh/test-linux-0.1.tbz"
          sha256 TEST_SHA256
        end
      end

      expect(f.stable.url).to eq("https://brew.sh/test-macos-0.1.tbz")
    end
  end

  describe "#on_macos" do
    it "adds a dependency on macos only" do
      f = formula do
        homepage "https://brew.sh"

        url "https://brew.sh/test-0.1.tbz"
        sha256 TEST_SHA256

        depends_on "hello_both"

        on_macos do
          depends_on "hello_macos"
        end

        on_linux do
          depends_on "hello_linux"
        end
      end

      expect(f.class.stable.deps[0].name).to eq("hello_both")
      expect(f.class.stable.deps[1].name).to eq("hello_macos")
      expect(f.class.stable.deps[2]).to eq(nil)
    end
  end

  describe "#on_macos" do
    it "adds a patch on macos only" do
      f = formula do
        homepage "https://brew.sh"

        url "https://brew.sh/test-0.1.tbz"
        sha256 TEST_SHA256

        patch do
          url "patch_both"
        end

        on_macos do
          patch do
            url "patch_macos"
          end
        end

        on_linux do
          patch do
            url "patch_linux"
          end
        end
      end

      expect(f.patchlist.length).to eq(2)
      expect(f.patchlist.first.strip).to eq(:p1)
      expect(f.patchlist.first.url).to eq("patch_both")
      expect(f.patchlist.second.strip).to eq(:p1)
      expect(f.patchlist.second.url).to eq("patch_macos")
    end
  end
end
